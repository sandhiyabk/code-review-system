# core/github_integration.py
"""
Safe, production-grade GitHub Integration for the Code Review Assistant.

Allows users to submit a GitHub file URL or repository file path
for review instead of manually pasting code. Uses the GitHub REST API
(NOT raw URL fetching) for proper validation, error handling, and
rate-limit awareness.

Security considerations:
- Only accepts github.com URLs (rejects raw/gitlab/bitbucket)
- Sanitizes file paths (rejects ../, ~/, //, null bytes)
- Never logs file content (may contain secrets)
- Optional GITHUB_TOKEN via env var for higher rate limits
"""

import os
import re
import base64
import time
from typing import Optional
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv
import requests

# Load env vars — GITHUB_TOKEN is optional
load_dotenv()


class GitHubIntegration:
    """
    GitHub repository file fetcher using the GitHub REST API.

    Supports two input modes:
    1. A full GitHub file URL:
       https://github.com/username/repo/blob/main/file.py
    2. Repository name + file path:
       owner="username", repo="repo", file_path="src/utils/helper.py"

    The class validates inputs, checks repository status, enforces
    file size/extension limits, and fetches content via the GitHub
    Contents API.
    """

    GITHUB_API_BASE = "https://api.github.com"

    # Only files up to 100KB are accepted to prevent
    # resource exhaustion and LLM token overflow
    MAX_FILE_SIZE_BYTES = 100_000

    # Whitelist of supported programming languages/extensions
    ALLOWED_EXTENSIONS = [
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"
    ]

    # Map extensions to language labels for display
    EXTENSION_TO_LANGUAGE = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
    }

    def __init__(self):
        """
        Initialize the GitHub client.

        Reads optional GITHUB_TOKEN from env vars for higher
        rate limits (5000/hr vs 60/hr unauthenticated).
        Never hardcodes tokens in source code.
        """
        self.token = os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()

        # If a token exists, send it as Bearer auth
        if self.token:
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
            })
        else:
            # Public repos work without auth
            self.session.headers.update({
                "Accept": "application/vnd.github+json",
            })

    # ──────────────────────────────────────────────────────
    # URL Parsing and Validation
    # ──────────────────────────────────────────────────────
    def parse_github_url(self, url: str) -> dict:
        """
        Parse and validate a GitHub file URL.

        Accepts URLs only of the form:
            https://github.com/{owner}/{repo}/blob/{branch}/{file_path}
        or with /tree/ (for browsing, but files are under blob/).

        Returns:
            {
                "owner": "user",
                "repo": "repo",
                "branch": "main",
                "file_path": "src/file.py",
                "extension": ".py"
            }

        Raises ValueError with a clear message for invalid URLs.
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty.")

        url = url.strip()

        # ── Validate domain is github.com ONLY ──
        # Reject: gitlab.com, bitbucket.org, raw.githubusercontent.com
        parsed = urlparse(url)

        # Normalize hostname (case-insensitive, strip www.)
        host = (parsed.hostname or "").lower()
        if host in ("www.github.com",):
            host = "github.com"

        if host != "github.com":
            # Provide specific feedback for common non-github domains
            if "githubusercontent" in host:
                raise ValueError(
                    "Raw GitHub URLs (githubusercontent.com) are not "
                    "allowed. Please use the regular github.com URL "
                    "for the file instead."
                )
            if host in ("gitlab.com", "bitbucket.org"):
                raise ValueError(
                    f"Only GitHub repositories are supported, not {host}."
                )
            raise ValueError(
                f"URL must be from github.com, got '{host}'."
            )

        # ── Reject double slashes in the URL path early ──
        # The split below filters empty segments, which would otherwise
        # silently hide a "//" traversal/malformation. Detect it here
        # on the raw path before any sanitization happens.
        if "//" in parsed.path:
            raise ValueError(
                "File path contains invalid double slashes (//)."
            )

        # ── Extract path components ──
        # Format: /owner/repo/blob/branch/path/to/file
        path_parts = [unquote(p) for p in parsed.path.split("/") if p]

        # Minimum: [owner, repo, blob, branch, file]
        if len(path_parts) < 4:
            raise ValueError(
                "Invalid GitHub URL format. Expected: "
                "github.com/owner/repo/blob/branch/path/file"
            )

        owner = path_parts[0]
        repo = path_parts[1]

        # Validate the ref marker (must be blob/ or tree/)
        # We do NOT accept raw/ or api/ URLs
        ref_marker = path_parts[2].lower() if len(path_parts) > 2 else ""
        if ref_marker not in ("blob", "tree"):
            raise ValueError(
                "URL must contain a 'blob' or 'tree' segment "
                "(e.g. github.com/user/repo/blob/main/file.py). "
                "Raw URLs and API URLs are not supported."
            )

        # Branch is the next segment, remaining is file path
        # Branch may contain slashes in some repos, but typically not.
        # We take path_parts[3] as branch and the rest as file path.
        if len(path_parts) < 5:
            raise ValueError(
                "URL does not contain a file path. "
                "Expected: github.com/owner/repo/blob/branch/path/file"
            )

        branch = path_parts[3]
        file_path = "/".join(path_parts[4:])

        # ── Validate owner/repo identifiers ──
        # Github usernames/repos use [A-Za-z0-9-] typically
        if not re.match(r"^[A-Za-z0-9-]+$", owner):
            raise ValueError(f"Invalid repository owner name: '{owner}'")
        if not re.match(r"^[A-Za-z0-9_.-]+$", repo):
            raise ValueError(f"Invalid repository name: '{repo}'")

        # ── Validate file extension is allowed ──
        extension = self._get_extension(file_path)
        if extension not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File extension '{extension}' is not supported. "
                f"Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )

        # ── Sanitize file_path against traversal attacks ──
        self._sanitize_path(file_path)

        return {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "file_path": file_path,
            "extension": extension,
            "url_type": ref_marker,
        }

    def _get_extension(self, file_path: str) -> str:
        """
        Extract the file extension from a file path.

        Returns '' if no extension is found (e.g. path ends
        in a directory or has no dot).
        """
        # Normalize path separators
        normalized = file_path.replace("\\", "/")
        filename = normalized.rsplit("/", 1)[-1]
        _, ext = os.path.splitext(filename)
        return ext.lower()

    def _sanitize_path(self, file_path: str) -> None:
        """
        Validate that the file path is safe.

        Rejects: directory traversal (../), home paths (~/),
        URL-encoded slashes, double slashes (//), null bytes.

        Raises ValueError if any unsafe pattern is detected.
        """
        # Reject null bytes immediately — these can break APIs
        if "\x00" in file_path:
            raise ValueError("File path contains invalid null byte.")

        # Decode URL-encoded patterns to catch hidden traversal
        decoded = unquote(file_path)

        # Block directory traversal: ../ or ../
        if re.search(r"\.\./", decoded):
            raise ValueError(
                "File path contains directory traversal (../). Not allowed."
            )

        # Block home directory references
        if "~/" in decoded or decoded.startswith("~"):
            raise ValueError(
                "File path must not reference home directory (~)."
            )

        # Block double slashes (could hide traversal or be malformed)
        if "//" in decoded:
            raise ValueError("File path contains invalid double slashes (//).")

        # Block backslash path separators (Windows traversal trick)
        if "\\" in decoded and ("..\\" in decoded or ".." in decoded.replace("\\", "")):
            # Allow single backslashes only if not traversal
            if re.search(r"\.\.\\", decoded):
                raise ValueError(
                    "File path contains directory traversal (..\\). Not allowed."
                )

        # Block any remaining suspicious characters
        # Allow only typical filename characters plus slashes, dots, underscores, hyphens
        if not re.match(r"^[A-Za-z0-9_./\-]+$", decoded):
            raise ValueError(
                "File path contains unsupported characters. "
                "Only letters, numbers, dots, slashes, dashes, "
                "and underscores are allowed."
            )

    # ──────────────────────────────────────────────────────
    # Repository Validation
    # ──────────────────────────────────────────────────────
    def validate_repository(self, owner: str, repo: str) -> dict:
        """
        Verify that a GitHub repository exists and is usable.

        Uses: GET /repos/{owner}/{repo}

        Checks:
        - Exists (404 = not found)
        - Is public (private repos require auth; we don't support)
        - Is not archived (warn user)

        Returns repository metadata dict.

        Handles rate-limit exhaustion gracefully.
        """
        # Make API call to validate the repository
        endpoint = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}"

        # Fetch repository metadata
        try:
            resp = self.session.get(endpoint, timeout=15)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to GitHub API. "
                "Please check your internet connection."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                "GitHub API request timed out. Please try again."
            )

        # ── Handle rate-limit warnings ──
        self._check_rate_limit_remaining(resp)

        # ── Handle HTTP errors ──
        if resp.status_code == 404:
            raise ValueError(
                f"Repository '{owner}/{repo}' not found on GitHub. "
                "Double-check the owner and repository name."
            )
        if resp.status_code == 403:
            # Could be rate-limited or access denied
            if "rate" in resp.text.lower():
                raise PermissionError(
                    "GitHub API rate limit exceeded. "
                    "Please try again later."
                )
            raise PermissionError(
                f"Access denied to '{owner}/{repo}'. "
                "This repository may be private or unavailable."
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"GitHub API returned error {resp.status_code}. "
                "Please try again."
            )

        # ── Parse repository metadata ──
        data = resp.json()

        # Check if private (we only support public repos)
        if data.get("private"):
            raise ValueError(
                f"Repository '{owner}/{repo}' is private. "
                "Only public repositories can be reviewed."
            )

        # Check if archived — warn but allow review
        if data.get("archived"):
            print(
                f"[github] NOTE: Repository '{owner}/{repo}' is archived. "
                "Content may be outdated."
            )
            return {
                "owner": owner,
                "repo": repo,
                "full_name": data.get("full_name"),
                "description": data.get("description"),
                "archived": True,
                "language": data.get("language"),
                "default_branch": data.get("default_branch", "main"),
                "html_url": data.get("html_url"),
            }

        return {
            "owner": owner,
            "repo": repo,
            "full_name": data.get("full_name"),
            "description": data.get("description"),
            "archived": False,
            "language": data.get("language"),
            "default_branch": data.get("default_branch", "main"),
            "html_url": data.get("html_url"),
        }

    def _check_rate_limit_remaining(self, resp) -> None:
        """
        Check X-RateLimit-Remaining header and warn if nearly exhausted.

        If remaining is very low (< 5), raise an error asking the
        user to wait. This prevents failing mid-request.
        """
        remaining_raw = resp.headers.get("X-RateLimit-Remaining")
        if remaining_raw is None:
            return  # Header not available (shouldn't happen)

        try:
            remaining = int(remaining_raw)
        except (ValueError, TypeError):
            return

        if remaining < 5:
            # Compute minutes until reset if we have the info
            reset_raw = resp.headers.get("X-RateLimit-Reset")
            minutes = 0
            if reset_raw:
                try:
                    reset_ts = int(reset_raw)
                    minutes = max(0, int((reset_ts - time.time()) / 60))
                except (ValueError, TypeError):
                    pass

            minutes_text = (
                f" Please try again in about {minutes} minute(s)."
                if minutes
                else " Please try again later."
            )
            raise PermissionError(
                "GitHub API rate limit nearly exceeded. "
                f"Only {remaining} request(s) remaining.{minutes_text}"
            )

    # ──────────────────────────────────────────────────────
    # File Content Fetching
    # ──────────────────────────────────────────────────────
    def fetch_file_content(
        self,
        owner: str,
        repo: str,
        file_path: str,
        branch: str = "main"
    ) -> dict:
        """
        Fetch file content via the GitHub Contents API.

        Uses: GET /repos/{owner}/{repo}/contents/{path}?ref={branch}

        Why Contents API (not raw URLs):
        - Validates file existence properly with status codes
        - Returns metadata (size, encoding, sha)
        - Respects GitHub rate limiting
        - Content comes base64-encoded → we decode safely

        Returns:
            {
                "content": "decoded file content as string",
                "file_name": "file.py",
                "file_size": 4521,
                "language": "python",
                "sha": "abc123",
                "html_url": "github.com/..."
            }
        """
        # ── Sanitize inputs before making API call ──
        self._sanitize_path(file_path)

        extension = self._get_extension(file_path)
        if extension not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File extension '{extension}' is not supported. "
                f"Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )

        # Validate the repo before fetching (catches 404/403 early)
        repo_info = self.validate_repository(owner, repo)

        # Use default branch if none provided
        if not branch:
            branch = repo_info.get("default_branch", "main")

        # ── Build Contents API endpoint ──
        encoded_path = "/".join(
            [requests.utils.quote(part, safe="")
             for part in file_path.split("/")]
        )
        endpoint = (
            f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}"
            f"/contents/{encoded_path}?ref={branch}"
        )

        # ── Make the request ──
        try:
            resp = self.session.get(endpoint, timeout=15)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to GitHub API. "
                "Please check your internet connection."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                "GitHub API request timed out. Please try again."
            )

        # Rate limit check
        self._check_rate_limit_remaining(resp)

        # ── Handle error responses ──
        if resp.status_code == 404:
            raise ValueError(
                f"File '{file_path}' not found on branch '{branch}' "
                f"in {owner}/{repo}. Double-check the path and branch."
            )
        if resp.status_code == 403:
            if "rate" in resp.text.lower():
                self._handle_rate_limit_error(resp)
            raise PermissionError(
                "Access denied. The file may be restricted or "
                "the API rate limit was exceeded."
            )
        if resp.status_code == 409:
            raise RuntimeError(
                f"Git repository is empty or in an unusable state: "
                f"{owner}/{repo}"
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"GitHub API returned error {resp.status_code} "
                f"while fetching the file."
            )

        # ── Parse the response ──
        data = resp.json()

        # If this is a directory listing (not a file), refuse
        if data.get("type") != "file":
            raise ValueError(
                f"'{file_path}' is a directory, not a file. "
                "Please provide a path to a specific file."
            )

        # ── Enforce file size limit ──
        file_size = data.get("size", 0)
        if file_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File is {file_size/1024:.1f}KB which exceeds the "
                f"{self.MAX_FILE_SIZE_BYTES/1024:.0f}KB limit. "
                "Please paste the specific function you want reviewed."
            )

        # ── Decode base64 content ──
        encoding = data.get("encoding", "base64")
        content_b64 = data.get("content", "")

        if encoding != "base64":
            raise ValueError(
                f"Unexpected encoding '{encoding}'. "
                "Cannot decode file content."
            )

        try:
            # Decode base64 string content
            content = base64.b64decode(content_b64)
        except (ValueError, TypeError) as e:
            raise ValueError("Could not decode file content.") from e

        # Check for binary file — try to decode as UTF-8 text
        try:
            content_str = content.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                "Binary files cannot be reviewed. "
                "Please provide a plain-text source file."
            )

        # Prevent printing content to logs (may contain secrets)
        file_name = file_path.rsplit("/", 1)[-1]
        print(
            f"[github] Fetched '{file_path}' from "
            f"{owner}/{repo}@{branch} — "
            f"{file_size} bytes"
        )

        return {
            "content": content_str,
            "file_name": file_name,
            "file_size": file_size,
            "language": self.EXTENSION_TO_LANGUAGE.get(extension, "unknown"),
            "sha": data.get("sha", ""),
            "html_url": data.get("html_url", ""),
            "download_url": data.get("download_url", ""),
            "extension": extension,
            "branch": branch,
        }

    # ──────────────────────────────────────────────────────
    # Main Public Entry Point
    # ──────────────────────────────────────────────────────
    def fetch_from_url(self, github_url: str) -> dict:
        """
        Main public method — fetch a file from a GitHub URL.

        Parses, validates, and fetches in one call.
        This is the only method the Streamlit UI needs to call.

        Returns the same structure as fetch_file_content plus
        the original URL for attribution.
        """
        # Parse and validate the URL
        parsed = self.parse_github_url(github_url)

        # Fetch the file content
        result = self.fetch_file_content(
            owner=parsed["owner"],
            repo=parsed["repo"],
            file_path=parsed["file_path"],
            branch=parsed["branch"],
        )

        # Add attribution
        result["original_url"] = github_url
        result["owner"] = parsed["owner"]
        result["repo"] = parsed["repo"]

        return result

    # ──────────────────────────────────────────────────────
    # Rate Limit Status
    # ──────────────────────────────────────────────────────
    def get_rate_limit_status(self) -> dict:
        """
        Check current GitHub API rate limit status.

        Uses: GET /rate_limit

        Returns:
            {
                "remaining": 45,
                "limit": 60,
                "reset_in_minutes": 23,
                "is_authenticated": False
            }
        """
        endpoint = f"{self.GITHUB_API_BASE}/rate_limit"

        try:
            resp = self.session.get(endpoint, timeout=15)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout):
            # If we can't reach the API, return unknown status
            return {
                "remaining": None,
                "limit": None,
                "reset_in_minutes": None,
                "is_authenticated": bool(self.token),
                "available": False,
            }

        if resp.status_code != 200:
            return {
                "remaining": None,
                "limit": None,
                "reset_in_minutes": None,
                "is_authenticated": bool(self.token),
                "available": False,
            }

        try:
            data = resp.json()
            # The rate_limit endpoint returns limits per resource
            # For unauthenticated, it's under 'core'
            core = data.get("resources", {}).get("core", {})
            remaining = core.get("remaining")
            limit = core.get("limit")
            reset_ts = core.get("reset", 0)

            reset_minutes = 0
            if reset_ts:
                reset_minutes = max(0, int((reset_ts - time.time()) / 60))

            return {
                "remaining": remaining,
                "limit": limit,
                "reset_in_minutes": reset_minutes,
                "is_authenticated": bool(self.token),
                "available": True,
            }
        except ValueError:
            return {
                "remaining": None,
                "limit": None,
                "reset_in_minutes": None,
                "is_authenticated": bool(self.token),
                "available": False,
            }

    def _handle_rate_limit_error(self, resp) -> None:
        """
        Parse rate-limit response to give a friendly error message.
        """
        reset_raw = resp.headers.get("X-RateLimit-Reset")
        minutes = 0
        if reset_raw:
            try:
                minutes = max(0, int((int(reset_raw) - time.time()) / 60))
            except (ValueError, TypeError):
                pass

        minutes_text = (
            f" Please try again in about {minutes} minute(s)."
            if minutes
            else ""
        )
        raise PermissionError(
            f"GitHub API rate limit reached.{minutes_text}"
        )
