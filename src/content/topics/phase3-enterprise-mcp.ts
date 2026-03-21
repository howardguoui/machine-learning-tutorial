import type { TopicContent } from '../types'

export const githubMCP: TopicContent = {
  id: 'github-mcp',
  title: { en: 'GitHub MCP Integration', zh: 'GitHub MCP 集成' },
  contentType: 'code',
  content: {
    en: `## GitHub MCP Integration

Expose GitHub operations as MCP tools so agents can search repos, read issues, and monitor pull requests — all from natural language queries.

### Setup

\`\`\`python
# phase3_enterprise/enterprise_mcp/github_workflows.py
import json, urllib.request
from mcp.server.fastmcp import FastMCP
from shared.config import settings

mcp = FastMCP("github-server")


def _github_headers() -> dict[str, str]:
    """Return GitHub API auth headers. Requires GITHUB_TOKEN in .env."""
    if not settings.github_token:
        raise ValueError("GITHUB_TOKEN not set in .env")
    return {
        "Authorization": f"token {settings.github_token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _github_get(path: str):
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url, headers=_github_headers())
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())
\`\`\`

### Tool Definitions

\`\`\`python
@mcp.tool()
def list_repos(username: str) -> str:
    """List public repositories for a GitHub user or organization."""
    repos = _github_get(f"/users/{username}/repos?sort=updated&per_page=10")
    result = [
        {"name": r["name"], "description": r["description"], "stars": r["stargazers_count"]}
        for r in repos
    ]
    return json.dumps(result, indent=2)


@mcp.tool()
def list_issues(owner: str, repo: str, state: str = "open") -> str:
    """List issues in a repository. state: 'open', 'closed', or 'all'."""
    issues = _github_get(f"/repos/{owner}/{repo}/issues?state={state}&per_page=10")
    result = [
        {
            "number": i["number"],
            "title": i["title"],
            "state": i["state"],
            "labels": [l["name"] for l in i.get("labels", [])],
        }
        for i in issues
        if "pull_request" not in i   # filter out PRs from issues list
    ]
    return json.dumps(result, indent=2)


@mcp.tool()
def get_issue(owner: str, repo: str, issue_number: int) -> str:
    """Get full details of a specific GitHub issue."""
    issue = _github_get(f"/repos/{owner}/{repo}/issues/{issue_number}")
    return json.dumps({
        "number": issue["number"],
        "title": issue["title"],
        "state": issue["state"],
        "body": issue["body"],
        "labels": [l["name"] for l in issue.get("labels", [])],
        "comments": issue["comments"],
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
\`\`\`

### Setup Requirements

\`\`\`bash
# .env
GITHUB_TOKEN=ghp_your_token_here   # classic token with 'repo' scope

# Run the server
python phase3_enterprise/enterprise_mcp/github_workflows.py
\`\`\`

### Agent Interaction Example

\`\`\`
User: "What open bugs exist in the howardguoui/machine-learning-tutorial repo?"

Agent Action: list_issues("howardguoui", "machine-learning-tutorial", "open")
Observation: [{"number": 3, "title": "Build error on phase3", "labels": ["bug"]}, ...]

Agent Action: get_issue("howardguoui", "machine-learning-tutorial", 3)
Observation: {"body": "Getting TS2305 error when importing hybrid_chain...", ...}

Final Answer: Issue #3 reports a TypeScript import error in the hybrid_chain module...
\`\`\``,

    zh: `## GitHub MCP 集成

将 GitHub 操作公开为 MCP 工具，以便代理可以搜索仓库、读取问题并监控拉取请求。

### 工具定义

\`\`\`python
@mcp.tool()
def list_repos(username: str) -> str:
    """列出 GitHub 用户或组织的公共仓库。"""
    repos = _github_get(f"/users/{username}/repos?sort=updated&per_page=10")
    return json.dumps([
        {"name": r["name"], "description": r["description"], "stars": r["stargazers_count"]}
        for r in repos
    ], indent=2)


@mcp.tool()
def list_issues(owner: str, repo: str, state: str = "open") -> str:
    """列出仓库中的问题。state：'open'、'closed' 或 'all'。"""
    issues = _github_get(f"/repos/{owner}/{repo}/issues?state={state}&per_page=10")
    return json.dumps([
        {"number": i["number"], "title": i["title"], "state": i["state"]}
        for i in issues if "pull_request" not in i
    ], indent=2)


@mcp.tool()
def get_issue(owner: str, repo: str, issue_number: int) -> str:
    """获取特定 GitHub 问题的完整详情。"""
    issue = _github_get(f"/repos/{owner}/{repo}/issues/{issue_number}")
    return json.dumps({
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
    }, indent=2)
\`\`\`

### 设置要求

\`\`\`bash
# .env
GITHUB_TOKEN=ghp_your_token_here   # 具有 'repo' 范围的经典令牌
\`\`\``,
  },
}

export const slackMCP: TopicContent = {
  id: 'slack-mcp',
  title: { en: 'Slack MCP Integration', zh: 'Slack MCP 集成' },
  contentType: 'code',
  content: {
    en: `## Slack MCP Integration

Expose Slack messaging as MCP tools so agents can send plain messages or formatted notifications to any channel.

### Setup

\`\`\`python
# phase3_enterprise/enterprise_mcp/slack_notifier.py
import json, urllib.request
from mcp.server.fastmcp import FastMCP
from shared.config import settings

mcp = FastMCP("slack-server")
SLACK_API_BASE = "https://slack.com/api"


def _slack_post(endpoint: str, payload: dict):
    """Post to the Slack Web API."""
    if not settings.slack_bot_token:
        raise ValueError("SLACK_BOT_TOKEN not set in .env")

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{SLACK_API_BASE}/{endpoint}",
        data=data,
        headers={
            "Authorization": f"Bearer {settings.slack_bot_token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())
\`\`\`

### Tool Definitions

\`\`\`python
@mcp.tool()
def send_message(channel: str, text: str) -> str:
    """Send a plain text message to a Slack channel.

    Args:
        channel: Channel name ("#general") or channel ID ("C0123456").
        text:    Message text (supports Slack markdown: *bold*, _italic_).
    """
    result = _slack_post("chat.postMessage", {"channel": channel, "text": text})
    if result.get("ok"):
        return f"✅ Message sent to {channel}"
    return f"❌ Slack error: {result.get('error')}"


@mcp.tool()
def send_notification(
    channel: str,
    title: str,
    message: str,
    color: str = "good",
) -> str:
    """Send a formatted notification with a coloured attachment.

    Args:
        color: "good" (green), "warning" (yellow), "danger" (red).
    """
    payload = {
        "channel": channel,
        "attachments": [{
            "color": color,
            "title": title,
            "text": message,
            "footer": "learn-llm agent",
        }],
    }
    result = _slack_post("chat.postMessage", payload)
    if result.get("ok"):
        return f"✅ Notification sent to {channel}: {title}"
    return f"❌ Slack error: {result.get('error')}"


if __name__ == "__main__":
    mcp.run()
\`\`\`

### Setup Requirements

\`\`\`bash
# .env
SLACK_BOT_TOKEN=xoxb-your-bot-token-here

# Required OAuth scopes for your Slack App:
# chat:write       — send messages
# channels:read    — list public channels
\`\`\`

### Agent Interaction Example

\`\`\`
User: "Notify #deployments that the RAG pipeline evaluation passed with 0.92 overall score"

Agent Action: send_notification(
    channel="#deployments",
    title="✅ RAG Evaluation Passed",
    message="Overall score: 0.92 | Precision: 0.95 | Faithfulness: 0.91 | Relevance: 0.90",
    color="good"
)
Observation: ✅ Notification sent to #deployments: ✅ RAG Evaluation Passed

Final Answer: I've sent the evaluation results to #deployments.
\`\`\`

### Notification Colors

| Color | Sidebar | Use case |
|---|---|---|
| \`"good"\` | Green | Success, passing tests |
| \`"warning"\` | Yellow | Degraded performance, warnings |
| \`"danger"\` | Red | Failures, errors, outages |`,

    zh: `## Slack MCP 集成

将 Slack 消息作为 MCP 工具公开，以便代理可以向任何频道发送纯文本消息或格式化通知。

### 工具定义

\`\`\`python
@mcp.tool()
def send_message(channel: str, text: str) -> str:
    """向 Slack 频道发送纯文本消息。"""
    result = _slack_post("chat.postMessage", {"channel": channel, "text": text})
    return f"✅ 消息已发送到 {channel}" if result.get("ok") else f"❌ Slack 错误：{result.get('error')}"


@mcp.tool()
def send_notification(channel: str, title: str, message: str, color: str = "good") -> str:
    """发送带有颜色附件的格式化通知。
    color：'good'（绿色）、'warning'（黄色）、'danger'（红色）
    """
    payload = {
        "channel": channel,
        "attachments": [{"color": color, "title": title, "text": message}],
    }
    result = _slack_post("chat.postMessage", payload)
    return f"✅ 通知已发送到 {channel}" if result.get("ok") else f"❌ Slack 错误"
\`\`\`

### 设置要求

\`\`\`bash
# .env
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
# 所需 OAuth 范围：chat:write、channels:read
\`\`\``,
  },
}
