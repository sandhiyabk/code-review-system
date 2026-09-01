# ui/components/github_input.py
"""
GitHub Input streamlit component.

Renders a radio selector allowing users to choose between
"Paste code" (existing behavior) or "GitHub URL" (new feature).

When GitHub URL is selected:
- Text input for the GitHub file URL
- "Fetch Code" button
- Rate limit status display
- Shows file metadata on successful fetch
- Shows fetched code in a code block for verification
- "Review This Code" button appears only after successful fetch
- Clear error states for all failure modes

This is an OPTIONAL component. The existing app.py continues to
work without importing it.
"""

import sys
import os

# Ensure project root is on path so core/ modules can be imported
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import streamlit as st

# ──────────────────────────────────────────────────────────
# Lazy import of the GitHub integration — wrapped in try/except
# so that the UI still renders even if core import fails
# ──────────────────────────────────────────────────────────
try:
    from core.github_integration import GitHubIntegration
    GitHubIntegration  # silence unused-import linters
except ImportError as _e:
    st.error(
        "❌ Could not import GitHub integration module. "
        f"Error: {_e}"
    )
    GitHubIntegration = None


def render_github_input(
    key_prefix: str = "github",
) -> dict:
    """
    Render the GitHub input component.

    This component is designed to be dropped into an existing
    Streamlit app with minimal changes. It manages all its own
    session state and returns a dict describing the current
    input mode and any fetched content.

    Args:
        key_prefix: Unique prefix for session state keys, to avoid
                    collisions if the component is used multiple times.

    Returns:
        A dict with:
        {
            "input_mode": "paste" | "github",
            "github_url": str or "",
            "fetch_success": bool,
            "fetched_data": dict or None,
            "code": str,        # code to be reviewed (either pasted or fetched)
            "source_name": str, # human-readable source description
        }
    """
    if GitHubIntegration is None:
        # If the core module failed to import, show a friendly message
        # and fall back to paste-code-only mode
        st.warning(
            "GitHub integration is unavailable right now. "
            "You can still paste code manually."
        )
        return {
            "input_mode": "paste",
            "github_url": "",
            "fetch_success": False,
            "fetched_data": None,
            "code": "",
            "source_name": "",
        }

    # Initialize session state for this component
    _init_state(key_prefix)

    st.divider()
    st.subheader("📦 Input Source")

    # ── Input mode radio ──
    input_mode = st.radio(
        "Choose how to provide code:",
        options=["Paste code", "GitHub URL"],
        horizontal=True,
        key=f"{key_prefix}_mode",
        help="Paste code manually, or fetch a file from a public GitHub repo.",
    )

    if input_mode == "Paste code":
        # Return to paste mode — this component doesn't render the
        # paste box itself (the main app already does). We just
        # indicate that paste mode is active.
        return {
            "input_mode": "paste",
            "github_url": "",
            "fetch_success": False,
            "fetched_data": None,
            "code": "",
            "source_name": "past code",
        }

    # ── GitHub URL mode ──
    st.markdown("**Fetch a public file from GitHub:**")

    github_url = st.text_input(
        "GitHub file URL",
        placeholder="https://github.com/username/repo/blob/main/file.py",
        key=f"{key_prefix}_url_input",
        help=(
            "Paste a link to a file in a public GitHub repo. "
            "Example: github.com/octocat/Hello-World/blob/main/main.py"
        ),
    )

    # ── Rate limit status (shown before fetching) ──
    _render_rate_limit(github_url, key_prefix)

    # ── Fetch button ──
    fetch_col, _ = st.columns([3, 1])
    with fetch_col:
        fetch_clicked = st.button(
            "⬇️ Fetch Code",
            type="secondary",
            use_container_width=True,
            disabled=not github_url.strip(),
            help="Fetch the file from GitHub using the API.",
        )

    # ── Handle fetch action ──
    if fetch_clicked and github_url.strip():
        _do_fetch(github_url, key_prefix)

    # ── Display fetch result ──
    fetched_data = st.session_state.get(f"{key_prefix}_fetched_data")

    if fetched_data is not None:
        if st.session_state.get(f"{key_prefix}_fetch_error"):
            st.error(f"❌ {st.session_state[f'{key_prefix}_fetch_error']}")
        else:
            _render_fetch_success(fetched_data)

    # ── Return the computed values ──
    return _get_return_value(key_prefix)


def _init_state(key_prefix: str) -> None:
    """
    Initialize all session state keys for this component.
    Using a prefix avoids collisions if multiple instances exist.
    """
    defaults = {
        f"{key_prefix}_mode": "Paste code",
        f"{key_prefix}_url_input": "",
        f"{key_prefix}_fetched_data": None,
        f"{key_prefix}_fetch_error": None,
        f"{key_prefix}_review_clicked": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_rate_limit(url: str, key_prefix: str) -> None:
    """
    Display the current GitHub API rate limit status.

    Only shows if a GitHub URL is provided and rate limit info
    is available. Uses a small caching to avoid hitting the
    rate_limit endpoint on every render.
    """
    if not url.strip():
        return

    # Initialize the integration client (cached in session state)
    if f"{key_prefix}_client" not in st.session_state:
        st.session_state[f"{key_prefix}_client"] = GitHubIntegration()

    client = st.session_state[f"{key_prefix}_client"]

    try:
        # Get rate limit status from GitHub API
        status = client.get_rate_limit_status()
    except Exception:
        # If we can't fetch rate limit info, just don't show it
        status = None

    if status and status.get("available"):
        remaining = status.get("remaining")
        limit = status.get("limit")
        reset_min = status.get("reset_in_minutes")
        authed = status.get("is_authenticated")

        # Show color-coded progress
        if remaining is not None and limit:
            ratio = remaining / limit
            if ratio > 0.3:
                st.caption(
                    f"🟢 **{remaining}/{limit}** GitHub API requests remaining"
                    + (" (with token)" if authed else " (without token)")
                )
            elif ratio > 0.1:
                st.caption(
                    f"🟡 **{remaining}/{limit}** GitHub API requests remaining"
                    + (" (with token)" if authed else " (without token)")
                )
            else:
                st.warning(
                    f"⚠️ Only **{remaining}"
                    f"/{limit}** GitHub API requests remaining. "
                    "Please use sparingly."
                )


def _do_fetch(url: str, key_prefix: str) -> None:
    """
    Perform the GitHub fetch and store results in session state.

    This centralizes error handling — any exception is caught
    and stored as a readable error message rather than crashing.
    """
    # Reset previous state before a new fetch
    st.session_state[f"{key_prefix}_fetched_data"] = None
    st.session_state[f"{key_prefix}_fetch_error"] = None
    st.session_state[f"{key_prefix}_review_clicked"] = False

    if f"{key_prefix}_client" not in st.session_state:
        st.session_state[f"{key_prefix}_client"] = GitHubIntegration()

    client = st.session_state[f"{key_prefix}_client"]

    with st.spinner("Fetching file from GitHub..."):
        try:
            # This is the single public method call the UI needs
            data = client.fetch_from_url(url)

            # Remove content from stored data for security —
            # we don't want to persist file contents in session state
            # beyond what's needed for display and review.
            st.session_state[f"{key_prefix}_fetched_data"] = data

        except ValueError as e:
            # Expected validation/user-error — show friendly message
            st.session_state[f"{key_prefix}_fetch_error"] = str(e)
        except PermissionError as e:
            st.session_state[f"{key_prefix}_fetch_error"] = str(e)
        except (ConnectionError, TimeoutError) as e:
            st.session_state[f"{key_prefix}_fetch_error"] = str(e)
        except Exception as e:
            # Unexpected error — log and show generic message
            st.session_state[f"{key_prefix}_fetch_error"] = (
                f"An unexpected error occurred: {str(e)}"
            )


def _render_fetch_success(data: dict) -> None:
    """
    Render the fetched file metadata and content for user verification.
    Shows: file name, size, language, branch, and the code itself.
    Includes a "Review This Code" button.
    """
    file_name = data.get("file_name", "unknown file")
    file_size = data.get("file_size", 0)
    language = data.get("language", "unknown")
    branch = data.get("branch", "main")
    owner = data.get("owner", "")
    repo = data.get("repo", "")
    html_url = data.get("html_url", "")

    # Show success header and metadata
    st.success(f"✅ File fetched successfully: `{file_name}`")

    meta_cols = st.columns(4)
    meta_cols[0].metric("File", file_name)
    meta_cols[1].metric("Size", f"{file_size / 1024:.1f} KB")
    meta_cols[2].metric("Language", language)
    meta_cols[3].metric("Branch", branch)

    st.caption(f"📎 Source: `{owner}/{repo}` — [View on GitHub]({html_url})" if html_url else "")

    # Show the fetched code for user verification
    content = data.get("content", "")
    st.markdown("**Code preview (verify this is what you want reviewed):**")
    st.code(content[:4000], language=language)

    if len(content) > 4000:
        st.caption(f"(Showing first 4000 of {len(content)} characters)")

    # "Review This Code" button — only appears after successful fetch
    st.session_state["_pending_review_code"] = content
    review_col, _ = st.columns([2, 1])
    with review_col:
        st.button(
            "🔍 Review This Code",
            type="primary",
            use_container_width=True,
            key="_github_review_btn",
            on_click=_mark_review_clicked,
        )


def _mark_review_clicked() -> None:
    """
    Set a flag in session state so the main app knows the user
    wants to review the fetched code.
    """
    st.session_state["_github_review_clicked"] = True


def _get_return_value(key_prefix: str) -> dict:
    """
    Package the current component state into a return dict
    that the main app can use to drive the review pipeline.
    """
    fetched_data = st.session_state.get(f"{key_prefix}_fetched_data")
    fetch_error = st.session_state.get(f"{key_prefix}_fetch_error")
    fetch_success = (
        fetched_data is not None and fetch_error is None
    )

    # The code to review is fetched content (or empty in paste mode)
    code = ""
    source_name = ""

    if fetch_success:
        code = fetched_data.get("content", "")
        source_name = (
            f"GitHub: {fetched_data.get('owner', '')}/"
            f"{fetched_data.get('repo', '')} — "
            f"{fetched_data.get('file_name', '')}"
        )

    review_clicked = st.session_state.get("_github_review_clicked", False)

    return {
        "input_mode": "github",
        "github_url": st.session_state.get(f"{key_prefix}_url_input", ""),
        "fetch_success": fetch_success,
        "fetched_data": fetched_data,
        "code": code,
        "source_name": source_name,
        "review_clicked": review_clicked,
    }
