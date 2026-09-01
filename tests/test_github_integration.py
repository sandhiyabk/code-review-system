# tests/test_github_integration.py
"""
Tests for the GitHub Integration module.

These tests validate:
- URL parsing with valid GitHub URLs
- URL parsing rejects non-GitHub URLs
- URL parsing rejects raw githubusercontent URLs
- File extension validation
- File path sanitization (traversal attacks)
- Repository validation (mocked API)
- File content fetching (mocked API)
- File size limit enforcement
- Rate limit status check (mocked)

All API calls are MOCKED — no real GitHub requests are made.
This keeps tests fast, reliable, and free of network dependence.

Run with:
    pytest tests/test_github_integration.py -v
"""

import sys
import os

# Add project root to path so core/ can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from core.github_integration import GitHubIntegration


@pytest.fixture
def gh():
    """
    Create a GitHubIntegration instance without a token.
    Setting GITHUB_TOKEN to empty avoids accidental use of a
    real token during tests.
    """
    os.environ["GITHUB_TOKEN"] = ""
    return GitHubIntegration()


# ══════════════════════════════════════════════════════════
# URL Parsing Tests
# ──────────────────────────────────────────────────────────

class TestURLParsing:
    """Test parse_github_url with various inputs."""

    def test_valid_github_url(self, gh):
        """A standard blob URL should parse correctly."""
        result = gh.parse_github_url(
            "https://github.com/user/repo/blob/main/src/utils/helper.py"
        )
        assert result["owner"] == "user"
        assert result["repo"] == "repo"
        assert result["branch"] == "main"
        assert result["file_path"] == "src/utils/helper.py"
        assert result["extension"] == ".py"

    def test_valid_url_with_python_extension(self, gh):
        """Python file URL should parse with correct extension."""
        result = gh.parse_github_url(
            "https://github.com/octocat/Hello-World/blob/master/main.py"
        )
        assert result["extension"] == ".py"
        assert result["owner"] == "octocat"
        assert result["repo"] == "Hello-World"

    def test_non_github_domain_rejected(self, gh):
        """gitlab.com URLs must be rejected with a clear message."""
        with pytest.raises(ValueError) as excinfo:
            gh.parse_github_url(
                "https://gitlab.com/user/repo/-/blob/main/file.py"
            )
        msg = str(excinfo.value)
        assert "github" in msg.lower()
        assert "gitlab" in msg.lower()

    def test_bitbucket_url_rejected(self, gh):
        """bitbucket.org URLs must be rejected with a clear message."""
        with pytest.raises(ValueError) as excinfo:
            gh.parse_github_url(
                "https://bitbucket.org/user/repo/src/master/file.py"
            )
        msg = str(excinfo.value)
        assert "github" in msg.lower()
        assert "bitbucket" in msg.lower()

    def test_raw_githubusercontent_rejected(self, gh):
        """
        Raw githubusercontent URLs must be rejected with a
        specific, helpful error message.
        """
        with pytest.raises(ValueError) as excinfo:
            gh.parse_github_url(
                "https://raw.githubusercontent.com/user/repo/main/file.py"
            )
        assert "Raw GitHub URLs" in str(excinfo.value)

    def test_www_github_works(self, gh):
        """www.github.com URLs should be accepted (normalized)."""
        result = gh.parse_github_url(
            "https://www.github.com/user/repo/blob/main/file.js"
        )
        assert result["owner"] == "user"
        assert result["extension"] == ".js"

    def test_non_blob_path_rejected(self, gh):
        """URLs without blob/tree segment must be rejected."""
        with pytest.raises(ValueError):
            gh.parse_github_url(
                "https://github.com/user/repo/raw/main/file.py"
            )

    def test_unsupported_extension_rejected(self, gh):
        """Files with disallowed extensions must be rejected."""
        with pytest.raises(ValueError) as excinfo:
            gh.parse_github_url(
                "https://github.com/user/repo/blob/main/README.md"
            )
        assert "extension" in str(excinfo.value).lower()

    def test_extension_case_insensitive(self, gh):
        """Uppercase extensions (.PY) should be accepted."""
        result = gh.parse_github_url(
            "https://github.com/user/repo/blob/main/File.PY"
        )
        assert result["extension"] == ".py"

    def test_empty_url_rejected(self, gh):
        """Empty URLs must be rejected."""
        with pytest.raises(ValueError):
            gh.parse_github_url("")


# ══════════════════════════════════════════════════════════
# File Path Sanitization Tests
# ──────────────────────────────────────────────────────────

class TestPathSanitization:
    """Test that unsafe file paths are rejected."""

    def test_directory_traversal_rejected(self, gh):
        """Paths with ../ must be rejected."""
        with pytest.raises(ValueError):
            gh.parse_github_url(
                "https://github.com/user/repo/blob/main/../secret.py"
            )

    def test_home_directory_rejected(self, gh):
        """Paths referencing home directory must be rejected."""
        with pytest.raises(ValueError):
            gh.parse_github_url(
                "https://github.com/user/repo/blob/main/~/home/.bashrc"
            )

    def test_double_slash_rejected(self, gh):
        """Paths with double slashes must be rejected."""
        with pytest.raises(ValueError):
            gh.parse_github_url(
                "https://github.com/user/repo/blob/main/src//file.py"
            )

    def test_null_byte_rejected(self, gh):
        """Paths with null bytes must be rejected."""
        with pytest.raises(ValueError):
            gh.parse_github_url(
                "https://github.com/user/repo/blob/main/file\x00.py"
            )


# ══════════════════════════════════════════════════════════
# Mocked API Tests for Validate / Fetch / Rate Limit
# These use monkeypatched responses so NO real GitHub API
# calls are made during tests.
# ──────────────────────────────────────────────────────────

class _FakeResponse:
    """A minimal fake response object for mocking requests."""

    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self.text = str(json_data) if json_data else ""

    def json(self):
        return self._json_data


class TestRepositoryValidation:
    """Test validate_repository with mocked GitHub responses."""

    def test_valid_public_repo(self, gh, monkeypatch):
        """A valid public repo should return metadata."""
        fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "description": "Test repo",
            "private": False,
            "archived": False,
            "language": "Python",
            "default_branch": "main",
            "html_url": "https://github.com/user/repo",
        }, headers={"X-RateLimit-Remaining": "50"})

        # Mock the session.get method
        monkeypatch.setattr(gh.session, "get", lambda *a, **k: fake)

        result = gh.validate_repository("user", "repo")
        assert result["owner"] == "user"
        assert result["repo"] == "repo"
        assert result["default_branch"] == "main"
        assert result["archived"] is False

    def test_not_found_repo(self, gh, monkeypatch):
        """A 404 should raise a ValueError with a clear message."""
        fake = _FakeResponse(404, None, headers={"X-RateLimit-Remaining": "50"})
        monkeypatch.setattr(gh.session, "get", lambda *a, **k: fake)

        with pytest.raises(ValueError) as excinfo:
            gh.validate_repository("nouser", "norepo")
        assert "not found" in str(excinfo.value).lower()

    def test_private_repo_rejected(self, gh, monkeypatch):
        """A private repo should be rejected gracefully."""
        fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": True,
            "archived": False,
        }, headers={"X-RateLimit-Remaining": "50"})
        monkeypatch.setattr(gh.session, "get", lambda *a, **k: fake)

        with pytest.raises(ValueError) as excinfo:
            gh.validate_repository("user", "repo")
        assert "private" in str(excinfo.value).lower()

    def test_archived_repo_warns(self, gh, monkeypatch):
        """An archived repo should still return metadata with a flag."""
        fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": False,
            "archived": True,
            "language": "Python",
            "default_branch": "main",
            "html_url": "https://github.com/user/repo",
        }, headers={"X-RateLimit-Remaining": "50"})
        monkeypatch.setattr(gh.session, "get", lambda *a, **k: fake)

        result = gh.validate_repository("user", "repo")
        assert result["archived"] is True


class TestRateLimitHandling:
    """Test rate-limit-related behaviors."""

    def test_rate_limit_remaining_ok(self, gh, monkeypatch):
        """Normal remaining requests should not raise."""
        fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": False,
            "archived": False,
            "default_branch": "main",
        }, headers={"X-RateLimit-Remaining": "50"})
        monkeypatch.setattr(gh.session, "get", lambda *a, **k: fake)

        # Should not raise
        result = gh.validate_repository("user", "repo")
        assert result["repo"] == "repo"

    def test_rate_limit_nearly_exceeded(self, gh, monkeypatch):
        """
        When remaining < 5, should raise a friendly PermissionError.
        """
        fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": False,
            "archived": False,
            "default_branch": "main",
        }, headers={"X-RateLimit-Remaining": "3"})
        monkeypatch.setattr(gh.session, "get", lambda *a, **k: fake)

        with pytest.raises(PermissionError) as excinfo:
            gh.validate_repository("user", "repo")
        assert "rate limit" in str(excinfo.value).lower()

    def test_get_rate_limit_status(self, gh, monkeypatch):
        """get_rate_limit_status should parse the rate_limit endpoint."""
        fake = _FakeResponse(200, {
            "resources": {
                "core": {
                    "remaining": 45,
                    "limit": 60,
                    "reset": 9999999999,  # far future
                }
            }
        })
        monkeypatch.setattr(gh.session, "get", lambda *a, **k: fake)

        status = gh.get_rate_limit_status()
        assert status["remaining"] == 45
        assert status["limit"] == 60
        assert status["is_authenticated"] is False
        assert status["available"] is True

    def test_get_rate_limit_authenticated(self, gh, monkeypatch):
        """
        If a token is set, is_authenticated should be True.
        """
        # Set a fake token
        gh.token = "fake-token"
        gh.session.headers.update({"Authorization": "Bearer fake-token"})

        fake = _FakeResponse(200, {
            "resources": {
                "core": {"remaining": 4500, "limit": 5000, "reset": 0}
            }
        })
        monkeypatch.setattr(gh.session, "get", lambda *a, **k: fake)

        status = gh.get_rate_limit_status()
        assert status["is_authenticated"] is True
        assert status["remaining"] == 4500


class TestFileContentFetching:
    """Test fetch_file_content with mocked GitHub responses."""

    def test_fetch_valid_file(self, gh, monkeypatch):
        """
        A valid base64-encoded file should be decoded correctly.
        """
        import base64

        content_b64 = base64.b64encode(
            b"def foo():\n    return 42\n"
        ).decode()

        # First call (validate_repository) returns repo info
        repo_fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": False,
            "archived": False,
            "default_branch": "main",
        }, headers={"X-RateLimit-Remaining": "50"})

        # Second call (fetch file) returns file content
        file_fake = _FakeResponse(200, {
            "type": "file",
            "size": 26,
            "encoding": "base64",
            "content": content_b64,
            "sha": "abc123",
            "html_url": "https://github.com/user/repo/blob/main/test.py",
        }, headers={"X-RateLimit-Remaining": "49"})

        # Use a counter to return different responses per call
        calls = {"n": 0}
        def mock_get(url, timeout=15):
            calls["n"] += 1
            return repo_fake if calls["n"] == 1 else file_fake

        monkeypatch.setattr(gh.session, "get", mock_get)

        result = gh.fetch_file_content("user", "repo", "test.py", "main")

        assert "def foo" in result["content"]
        assert result["file_name"] == "test.py"
        assert result["file_size"] == 26
        assert result["language"] == "python"
        assert result["sha"] == "abc123"

    def test_file_too_large_rejected(self, gh, monkeypatch):
        """
        Files larger than MAX_FILE_SIZE_BYTES must be rejected.
        """
        repo_fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": False,
            "archived": False,
            "default_branch": "main",
        }, headers={"X-RateLimit-Remaining": "50"})

        # Size > 100000
        file_fake = _FakeResponse(200, {
            "type": "file",
            "size": GitHubIntegration.MAX_FILE_SIZE_BYTES + 1,
            "encoding": "base64",
            "content": "",
        }, headers={"X-RateLimit-Remaining": "49"})

        calls = {"n": 0}
        def mock_get(url, timeout=15):
            calls["n"] += 1
            return repo_fake if calls["n"] == 1 else file_fake

        monkeypatch.setattr(gh.session, "get", mock_get)

        with pytest.raises(ValueError) as excinfo:
            gh.fetch_file_content("user", "repo", "big.py", "main")
        assert "limit" in str(excinfo.value).lower()

    def test_directory_rejected(self, gh, monkeypatch):
        """A directory entry must be rejected."""
        repo_fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": False,
            "archived": False,
            "default_branch": "main",
        }, headers={"X-RateLimit-Remaining": "50"})

        file_fake = _FakeResponse(200, {
            "type": "dir",  # not a file
            "size": 0,
        }, headers={"X-RateLimit-Remaining": "49"})

        calls = {"n": 0}
        def mock_get(url, timeout=15):
            calls["n"] += 1
            return repo_fake if calls["n"] == 1 else file_fake

        monkeypatch.setattr(gh.session, "get", mock_get)

        with pytest.raises(ValueError):
            gh.fetch_file_content("user", "repo", "src", "main")

    def test_file_not_found(self, gh, monkeypatch):
        """A 404 on content fetch should give a clear message."""
        repo_fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": False,
            "archived": False,
            "default_branch": "main",
        }, headers={"X-RateLimit-Remaining": "50"})

        file_fake = _FakeResponse(
            404, None, headers={"X-RateLimit-Remaining": "49"}
        )

        calls = {"n": 0}
        def mock_get(url, timeout=15):
            calls["n"] += 1
            return repo_fake if calls["n"] == 1 else file_fake

        monkeypatch.setattr(gh.session, "get", mock_get)

        with pytest.raises(ValueError) as excinfo:
            gh.fetch_file_content("user", "repo", "missing.py", "main")
        assert "not found" in str(excinfo.value).lower()

    def test_binary_file_rejected(self, gh, monkeypatch):
        """
        Base64 data that decodes to non-UTF8 (binary) should be rejected.
        """
        import base64

        # Some arbitrary binary bytes that aren't valid UTF-8 text
        content_b64 = base64.b64encode(b"\xff\xfe\x00\x01binary").decode()

        repo_fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": False,
            "archived": False,
            "default_branch": "main",
        }, headers={"X-RateLimit-Remaining": "50"})

        file_fake = _FakeResponse(200, {
            "type": "file",
            "size": 8,
            "encoding": "base64",
            "content": content_b64,
        }, headers={"X-RateLimit-Remaining": "49"})

        calls = {"n": 0}
        def mock_get(url, timeout=15):
            calls["n"] += 1
            return repo_fake if calls["n"] == 1 else file_fake

        monkeypatch.setattr(gh.session, "get", mock_get)

        with pytest.raises(ValueError) as excinfo:
            gh.fetch_file_content("user", "repo", "bin.py", "main")
        assert "binary" in str(excinfo.value).lower()


class TestFetchFromURL:
    """Test the full fetch_from_url public entry point."""

    def test_fetch_from_url(self, gh, monkeypatch):
        """
        fetch_from_url should parse, validate, and fetch in one call.
        """
        import base64

        content_b64 = base64.b64encode(b"print('hello')\n").decode()

        repo_fake = _FakeResponse(200, {
            "full_name": "user/repo",
            "private": False,
            "archived": False,
            "default_branch": "main",
        }, headers={"X-RateLimit-Remaining": "50"})

        file_fake = _FakeResponse(200, {
            "type": "file",
            "size": 16,
            "encoding": "base64",
            "content": content_b64,
            "sha": "def456",
            "html_url": "https://github.com/user/repo/blob/main/hello.py",
        }, headers={"X-RateLimit-Remaining": "49"})

        calls = {"n": 0}
        def mock_get(url, timeout=15):
            calls["n"] += 1
            return repo_fake if calls["n"] == 1 else file_fake

        monkeypatch.setattr(gh.session, "get", mock_get)

        result = gh.fetch_from_url(
            "https://github.com/user/repo/blob/main/hello.py"
        )

        assert "hello" in result["content"]
        assert result["file_name"] == "hello.py"
        assert result["original_url"].endswith("hello.py")
        assert result["owner"] == "user"
        assert result["repo"] == "repo"
