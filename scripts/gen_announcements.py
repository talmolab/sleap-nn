import re
import requests
import mkdocs_gen_files
import os
from datetime import datetime

OWNER = "talmolab"
REPOS = {
    "sleap": {"name": "SLEAP", "description": "Main GUI/application package"},
    "sleap-io": {"name": "sleap-io", "description": "I/O utilities package"},
    "sleap-nn": {"name": "sleap-nn", "description": "Neural network backend"},
}

GH_TOKEN = os.environ.get("GH_TOKEN", None)
if GH_TOKEN is None:
    GH_TOKEN = os.environ.get("GH_TOKEN_READ_ONLY", None)
if GH_TOKEN is None:
    GH_TOKEN = os.environ.get("GITHUB_TOKEN", None)
if GH_TOKEN is None:
    import subprocess

    try:
        proc = subprocess.run("gh auth token", shell=True, capture_output=True)
        GH_TOKEN = proc.stdout.decode().strip()
    except:
        GH_TOKEN = None

if GH_TOKEN is None:
    print("Warning: No GitHub token found, rate limits may be exceeded.")


def fetch_latest_releases(owner, repo, github_token=None, num_releases=5):
    """
    Fetches the latest release information from a GitHub repository.

    Parameters:
    - owner (str): The owner of the repository.
    - repo (str): The name of the repository.
    - github_token (str, optional): GitHub personal access token for authentication.
    - num_releases (int): Number of recent releases to fetch (default: 5).

    Returns:
    - dict: Contains "latest" (latest release) and "recent" (list of recent releases).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"

    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    headers["Accept"] = "application/vnd.github.v3+json"

    try:
        response = requests.get(url, headers=headers, params={"per_page": num_releases})

        if response.status_code != 200:
            print(f"Warning: Failed to fetch releases for {repo}: {response.status_code}")
            return {"latest": None, "recent": []}

        releases = response.json()

        if not releases:
            return {"latest": None, "recent": []}

        latest = releases[0] if releases else None
        recent = releases[:num_releases]

        return {
            "latest": {
                "tag_name": latest["tag_name"],
                "published_at": latest.get("published_at", ""),
                "html_url": latest["html_url"],
                "body": latest.get("body", ""),
            }
            if latest
            else None,
            "recent": [
                {
                    "tag_name": r["tag_name"],
                    "published_at": r.get("published_at", ""),
                    "html_url": r["html_url"],
                    "body": r.get("body", ""),
                }
                for r in recent
            ],
        }
    except Exception as e:
        print(f"Error fetching releases for {repo}: {e}")
        return {"latest": None, "recent": []}


def format_date(date_str):
    """Format ISO date string to readable format."""
    if not date_str:
        return ""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except:
        return date_str


def truncate_release_body(body, max_length=300):
    """Truncate release body to max_length characters."""
    if not body:
        return ""
    if len(body) <= max_length:
        return body
    # Try to truncate at a sentence boundary
    truncated = body[:max_length]
    last_period = truncated.rfind(".")
    if last_period > max_length * 0.7:  # Only use if we're not too far from the end
        return truncated[: last_period + 1] + "..."
    return truncated + "..."


# Fetch release information for all packages
package_releases = {}
for repo_key, repo_info in REPOS.items():
    print(f"Fetching releases for {repo_key}...")
    package_releases[repo_key] = fetch_latest_releases(OWNER, repo_key, GH_TOKEN, num_releases=1)

# Generate the announcements page
with mkdocs_gen_files.open("announcements.md", "w") as page:
    contents = [
        "# Announcements & Updates\n\n",
        "Stay informed about the latest releases across the SLEAP ecosystem. "
        "Important fixes or changes may land in `sleap-io` or `sleap-nn` independently of `sleap`, "
        "and these can critically affect SLEAP workflows (I/O, training, inference, etc.).\n\n",
        "---\n\n",
        "## Latest Versions\n\n",
    ]

    # Latest versions section 
    for repo_key, repo_info in REPOS.items():
        releases = package_releases[repo_key]
        latest = releases["latest"]

        if latest:
            version = latest["tag_name"]
            date = format_date(latest["published_at"])
            url = latest["html_url"]
            contents.append(
                f"**{repo_info['name']}**\n\n"
                f"**Latest Version:** [{version}]({url})"
            )
            if date:
                contents.append(f" (released {date})")
            contents.append(f"\n\n")
        else:
            contents.append(f"**{repo_info['name']}**\n\n")
            contents.append(f"*No releases found*\n\n")

    contents.append("---\n\n")
    contents.append("## Recent Updates\n\n")

    # Recent updates section 
    for repo_key, repo_info in REPOS.items():
        releases = package_releases[repo_key]
        recent = releases["recent"]

        contents.append(f"### {repo_info['name']}\n\n")
        
        if recent and len(recent) > 0:
            release = recent[0]  # Get only the first (latest) release
            version = release["tag_name"]
            date = format_date(release["published_at"])
            url = release["html_url"]
            body = release["body"]

            contents.append(f"**[{version}]({url})**")
            if date:
                contents.append(f" - {date}")
            contents.append("\n\n")

            if body:
                # Clean up the body - remove markdown headers that might interfere with TOC
                body_lines = body.split("\n")
                cleaned_body = []
                for line in body_lines[:20]:  # Limit to first 20 lines
                    # Skip markdown headers (##, ###, etc.) that would create TOC entries
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        # Skip headers like "## What's Changed", "### Summary", etc.
                        continue
                    cleaned_body.append(line)

                body_text = "\n".join(cleaned_body)
                truncated = truncate_release_body(body_text, max_length=400)
                contents.append(f"{truncated}\n\n")
                contents.append(f"[View full release notes →]({url})\n\n")
        else:
            contents.append("*No recent releases found*\n\n")

    contents.append("---\n\n")
    contents.append("## Upgrade Instructions\n\n")
    contents.append(
        "Use the commands below to upgrade to the latest versions based on your installation method.\n\n"
    )

    # Upgrade commands section with tabs
    contents.append('=== "uv tool install (System-wide)"\n')
    contents.append("    ```bash\n")
    contents.append("    # Windows/Linux (CUDA 12.8)\n")
    contents.append(
        "    uv tool install --upgrade sleap-nn[torch] --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple\n"
    )
    contents.append("    \n")
    contents.append("    # Windows/Linux (CUDA 11.8)\n")
    contents.append(
        "    uv tool install --upgrade sleap-nn[torch] --index https://download.pytorch.org/whl/cu118 --index https://pypi.org/simple\n"
    )
    contents.append("    \n")
    contents.append("    # Windows/Linux (CPU)\n")
    contents.append(
        "    uv tool install --upgrade sleap-nn[torch] --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple\n"
    )
    contents.append("    \n")
    contents.append("    # macOS\n")
    contents.append('    uv tool install --upgrade "sleap-nn[torch]"\n')
    contents.append("    ```\n\n")

    contents.append('=== "uvx (One-off commands)"\n')
    contents.append("    ```bash\n")
    contents.append(
        "    # uvx automatically uses the latest version. To force upgrade:\n"
    )
    contents.append("    # Windows/Linux (CUDA)\n")
    contents.append(
        "    uvx --upgrade --from \"sleap-nn[torch]\" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple sleap-nn ...\n"
    )
    contents.append("    \n")
    contents.append("    # Windows/Linux (CPU)\n")
    contents.append(
        "    uvx --upgrade --from \"sleap-nn[torch]\" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple sleap-nn ...\n"
    )
    contents.append("    \n")
    contents.append("    # macOS\n")
    contents.append('    uvx --upgrade "sleap-nn[torch]" ...\n')
    contents.append("    ```\n\n")

    contents.append('=== "uv add (Project-specific)"\n')
    contents.append("    ```bash\n")
    contents.append("    # Windows/Linux (CUDA 12.8)\n")
    contents.append(
        "    uv add --upgrade sleap-nn[torch] --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple\n"
    )
    contents.append("    \n")
    contents.append("    # Windows/Linux (CUDA 11.8)\n")
    contents.append(
        "    uv add --upgrade sleap-nn[torch] --index https://download.pytorch.org/whl/cu118 --index https://pypi.org/simple\n"
    )
    contents.append("    \n")
    contents.append("    # Windows/Linux (CPU)\n")
    contents.append(
        "    uv add --upgrade sleap-nn[torch] --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple\n"
    )
    contents.append("    \n")
    contents.append("    # macOS\n")
    contents.append('    uv add --upgrade "sleap-nn[torch]"\n')
    contents.append("    ```\n\n")

    contents.append('=== "pip (Conda environments)"\n')
    contents.append("    ```bash\n")
    contents.append("    # Windows/Linux (CUDA 12.8)\n")
    contents.append(
        "    pip install --upgrade sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128\n"
    )
    contents.append("    \n")
    contents.append("    # Windows/Linux (CUDA 11.8)\n")
    contents.append(
        "    pip install --upgrade sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118\n"
    )
    contents.append("    \n")
    contents.append("    # Windows/Linux (CPU)\n")
    contents.append(
        "    pip install --upgrade sleap-nn[torch] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu\n"
    )
    contents.append("    \n")
    contents.append("    # macOS\n")
    contents.append('    pip install --upgrade "sleap-nn[torch]"\n')
    contents.append("    ```\n\n")

    contents.append('=== "uv sync (From source)"\n')
    contents.append("    ```bash\n")
    contents.append("    # Upgrade all dependencies including sleap-nn\n")
    contents.append("    uv sync --upgrade --extra dev\n")
    contents.append("    \n")
    contents.append("    # Or for specific extras:\n")
    contents.append("    # Windows/Linux (CUDA 11.8)\n")
    contents.append("    uv sync --upgrade --extra dev --extra torch-cuda118\n")
    contents.append("    \n")
    contents.append("    # Windows/Linux (CUDA 12.8)\n")
    contents.append("    uv sync --upgrade --extra dev --extra torch-cuda128\n")
    contents.append("    \n")
    contents.append("    # macOS/CPU Only\n")
    contents.append("    uv sync --upgrade --extra dev --extra torch-cpu\n")
    contents.append("    ```\n\n")

    contents.append("---\n\n")
    contents.append(
        '!!! tip "Note"\n'
        "    For more detailed installation instructions, see the [Installation Guide](installation.md).\n"
    )

    page.writelines(contents)
