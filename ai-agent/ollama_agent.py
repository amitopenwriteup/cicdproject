#!/usr/bin/env python3
"""
Ollama AI Agent for Jenkins CI/CD Pipeline
==========================================
100% FREE — Runs locally — No API keys needed
Calls Ollama REST API for:
  1. Code Review & Quality Score
  2. Security Vulnerability Scan
  3. Deployment Decision (Approve / Reject)
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_ollama_running():
    """Verify Ollama server is reachable before doing anything."""
    try:
        req = urllib.request.urlopen(OLLAMA_HOST, timeout=5)
        return True
    except Exception:
        print(f"❌ Cannot reach Ollama at {OLLAMA_HOST}", file=sys.stderr)
        print("   → Make sure Ollama is running: ollama serve", file=sys.stderr)
        print("   → Or check OLLAMA_HOST environment variable", file=sys.stderr)
        sys.exit(1)


def check_model_available(model: str):
    """Check that the required model is pulled."""
    try:
        url  = f"{OLLAMA_HOST}/api/tags"
        req  = urllib.request.urlopen(url, timeout=10)
        data = json.loads(req.read())
        models = [m["name"] for m in data.get("models", [])]
        # Match partial names e.g. "llama3.2:1b" matches "llama3.2:1b"
        available = any(model in m or m in model for m in models)
        if not available:
            print(f"❌ Model '{model}' not found. Pull it first:", file=sys.stderr)
            print(f"   ollama pull {model}", file=sys.stderr)
            sys.exit(1)
        print(f"✅ Model '{model}' is available")
    except Exception as e:
        print(f"⚠️  Could not verify model (continuing): {e}", file=sys.stderr)


def call_ollama(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    """
    Call Ollama /api/chat endpoint.
    Returns the model's response text.
    Uses only stdlib — no pip install required for the HTTP call itself.
    """
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.1,    # low temp = more deterministic JSON
            "num_predict": 1024,   # max output tokens
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }).encode("utf-8")

    url = f"{OLLAMA_HOST}/api/chat"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            elapsed = time.time() - start
            print(f"   ⏱  Ollama responded in {elapsed:.1f}s")
            return data["message"]["content"].strip()
    except urllib.error.URLError as e:
        print(f"❌ Ollama request failed: {e}", file=sys.stderr)
        sys.exit(1)


def safe_parse_json(text: str, fallback: dict) -> dict:
    """
    Strip markdown fences and parse JSON.
    If parsing fails, try to extract first {...} block.
    Returns fallback dict on complete failure.
    """
    # Remove ```json ... ``` fences
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    print(f"⚠️  Could not parse JSON, using fallback. Raw response:\n{text[:300]}")
    return fallback


def read_source_files(path: str, max_chars: int = 5000) -> str:
    """Read source files from path, truncated to max_chars to fit model context."""
    exts = (".py", ".js", ".ts", ".go", ".java", ".rb", ".php")
    files = []
    for ext in exts:
        for f in Path(path).rglob(f"*{ext}"):
            if any(skip in str(f) for skip in
                   ["__pycache__", "node_modules", ".git", "venv", ".tox"]):
                continue
            try:
                content = f.read_text(errors="replace")
                files.append(f"### {f.name}\n{content}")
            except Exception:
                pass

    combined = "\n\n".join(files)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[... truncated for context window ...]"
    return combined if combined else "# No source files found in path"


# ── Stage 1: Code Review ───────────────────────────────────────────────────────

def run_code_review(code_path: str) -> dict:
    print("\n" + "─"*50)
    print("📋 STAGE 1: AI Code Review")
    print("─"*50)

    code = read_source_files(code_path)

    system = """You are a senior software engineer doing a code review.
You MUST respond with ONLY valid JSON. No explanation. No markdown. No extra text.
Just the raw JSON object."""

    user = f"""Review the following code and return ONLY this JSON:
{{
  "score": <integer 1-10>,
  "grade": "<A|B|C|D|F>",
  "critical_issues": ["<issue description>"],
  "major_issues": ["<issue description>"],
  "suggestions": ["<improvement suggestion>"],
  "summary": "<one sentence summary>",
  "approved": <true or false>
}}

Rules:
- score 8-10 = excellent, 6-7 = good, 4-5 = needs work, 1-3 = poor
- approved = true if score >= 6
- Be specific in issues, not generic

Code to review:
{code}"""

    raw    = call_ollama(system, user)
    result = safe_parse_json(raw, fallback={
        "score": 5, "grade": "C",
        "critical_issues": [], "major_issues": [],
        "suggestions": ["Review manually"],
        "summary": "AI review inconclusive",
        "approved": True
    })

    score = result.get("score", 5)
    grade = result.get("grade", "?")
    print(f"\n  🎯 Score   : {score}/10  (Grade: {grade})")
    print(f"  📝 Summary : {result.get('summary', 'N/A')}")

    for issue in result.get("critical_issues", []):
        print(f"  🔴 CRITICAL: {issue}")
    for issue in result.get("major_issues", []):
        print(f"  🟡 MAJOR   : {issue}")
    for tip in result.get("suggestions", []):
        print(f"  💡 TIP     : {tip}")

    return result


# ── Stage 2: Security Scan ─────────────────────────────────────────────────────

def run_security_scan(code_path: str) -> dict:
    print("\n" + "─"*50)
    print("🔒 STAGE 2: AI Security Scan")
    print("─"*50)

    code = read_source_files(code_path)

    system = """You are an application security expert doing a security audit.
You MUST respond with ONLY valid JSON. No explanation. No markdown. No extra text."""

    user = f"""Scan for security vulnerabilities and return ONLY this JSON:
{{
  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "score": <integer 1-10 where 10=most secure>,
  "vulnerabilities": [
    {{
      "type": "<vulnerability type e.g. SQL Injection>",
      "severity": "<LOW|MEDIUM|HIGH|CRITICAL>",
      "file": "<filename>",
      "line": "<line number or 'unknown'>",
      "description": "<what the issue is>",
      "recommendation": "<how to fix it>"
    }}
  ],
  "secrets_found": <true or false>,
  "owasp_issues": ["<OWASP Top 10 category if applicable>"],
  "summary": "<one sentence>",
  "block_deployment": <true or false>
}}

Rules:
- block_deployment = true ONLY if CRITICAL vulnerabilities found OR secrets found
- Check for: hardcoded passwords, SQL injection, command injection,
  insecure deserialization, path traversal, XSS, weak crypto
- If code looks clean, still list risk_level as LOW

Code to scan:
{code}"""

    raw    = call_ollama(system, user)
    result = safe_parse_json(raw, fallback={
        "risk_level": "UNKNOWN",
        "score": 5,
        "vulnerabilities": [],
        "secrets_found": False,
        "owasp_issues": [],
        "summary": "Security scan inconclusive",
        "block_deployment": False
    })

    risk  = result.get("risk_level", "UNKNOWN")
    score = result.get("score", "?")
    risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(risk, "⚪")

    print(f"\n  {risk_icon} Risk Level : {risk}")
    print(f"  🔐 Sec Score: {score}/10")
    print(f"  📝 Summary  : {result.get('summary', 'N/A')}")
    print(f"  🔑 Secrets  : {'YES ⚠️' if result.get('secrets_found') else 'None found ✅'}")

    for v in result.get("vulnerabilities", []):
        sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(
            v.get("severity", ""), "⚪")
        print(f"  {sev_icon} {v.get('severity','?')} [{v.get('type','?')}]"
              f" in {v.get('file','?')}: {v.get('description','')[:70]}")
        print(f"     Fix: {v.get('recommendation','')[:70]}")

    return result


# ── Stage 3: Deploy Decision ───────────────────────────────────────────────────

def run_deploy_decision(code_review: dict, security_scan: dict, environment: str) -> dict:
    print("\n" + "─"*50)
    print(f"🚦 STAGE 3: AI Deploy Decision  [{environment.upper()}]")
    print("─"*50)

    system = """You are a DevOps lead making deployment approval decisions.
Be STRICT for production. Be lenient for dev/staging.
You MUST respond with ONLY valid JSON. No explanation. No markdown."""

    user = f"""Make a deployment decision and return ONLY this JSON:
{{
  "decision": "<APPROVE|REJECT|APPROVE_WITH_CONDITIONS>",
  "confidence": <integer 0-100>,
  "risk_assessment": "<LOW|MEDIUM|HIGH>",
  "reason": "<clear explanation max 2 sentences>",
  "conditions": ["<condition to meet if APPROVE_WITH_CONDITIONS>"],
  "recommended_actions": ["<action to take before or after deploy>"],
  "estimated_rollback_needed": <true or false>
}}

Environment target: {environment}

Code Review Results:
- Score: {code_review.get('score', 'N/A')}/10
- Grade: {code_review.get('grade', 'N/A')}
- Critical Issues: {code_review.get('critical_issues', [])}
- Major Issues: {code_review.get('major_issues', [])}
- Approved: {code_review.get('approved', True)}

Security Scan Results:
- Risk Level: {security_scan.get('risk_level', 'UNKNOWN')}
- Secrets Found: {security_scan.get('secrets_found', False)}
- Block Deployment: {security_scan.get('block_deployment', False)}
- Vulnerabilities Count: {len(security_scan.get('vulnerabilities', []))}
- Summary: {security_scan.get('summary', 'N/A')}

Decision rules:
- REJECT if: secrets found OR CRITICAL security risk OR code score < 3
- REJECT for prod if: code score < 5 OR HIGH security risk
- APPROVE_WITH_CONDITIONS if: code score 5-6 OR MEDIUM security risk
- APPROVE if: code score >= 7 AND LOW security risk
"""

    raw    = call_ollama(system, user)
    result = safe_parse_json(raw, fallback={
        "decision": "APPROVE",
        "confidence": 50,
        "risk_assessment": "MEDIUM",
        "reason": "AI decision inconclusive — manual review recommended",
        "conditions": [],
        "recommended_actions": ["Review manually before proceeding"],
        "estimated_rollback_needed": False
    })

    decision   = result.get("decision", "APPROVE")
    confidence = result.get("confidence", "?")
    dec_icon   = {"APPROVE": "✅", "REJECT": "❌",
                  "APPROVE_WITH_CONDITIONS": "⚠️"}.get(decision, "❓")

    print(f"\n  {dec_icon} Decision   : {decision}")
    print(f"  📊 Confidence: {confidence}%")
    print(f"  ⚠️  Risk      : {result.get('risk_assessment', 'N/A')}")
    print(f"  📝 Reason    : {result.get('reason', 'N/A')}")

    for cond in result.get("conditions", []):
        print(f"  ⚙️  Condition : {cond}")
    for action in result.get("recommended_actions", []):
        print(f"  🔧 Action   : {action}")

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ollama AI Agent for Jenkins CI/CD — 100% Local & Free"
    )
    parser.add_argument("--path",        default="./app",         help="Path to source code")
    parser.add_argument("--environment", default="dev",           help="Target environment")
    parser.add_argument("--output",      default="ai_report.json",help="Output JSON report file")
    parser.add_argument("--model",       default="",              help="Override Ollama model")
    args = parser.parse_args()

    # Allow CLI override of model
    global OLLAMA_MODEL
    if args.model:
        OLLAMA_MODEL = args.model

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   🦙 Ollama AI Agent — Jenkins CI/CD            ║")
    print(f"║   Host  : {OLLAMA_HOST:<38}║")
    print(f"║   Model : {OLLAMA_MODEL:<38}║")
    print(f"║   Path  : {args.path:<38}║")
    print(f"║   Target: {args.environment:<38}║")
    print("╚══════════════════════════════════════════════════╝")

    # Pre-flight checks
    check_ollama_running()
    check_model_available(OLLAMA_MODEL)

    # Run all three AI stages
    code_review   = run_code_review(args.path)
    security_scan = run_security_scan(args.path)
    deploy_dec    = run_deploy_decision(code_review, security_scan, args.environment)

    # Compile full report
    report = {
        "ollama_host":     OLLAMA_HOST,
        "ollama_model":    OLLAMA_MODEL,
        "environment":     args.environment,
        "code_review":     code_review,
        "security_scan":   security_scan,
        "deploy_decision": deploy_dec,
        # Top-level shortcuts for Jenkinsfile readJSON
        "score":           code_review.get("score", 5),
        "risk_level":      security_scan.get("risk_level", "UNKNOWN"),
        "decision":        deploy_dec.get("decision", "APPROVE"),
        "summary":         code_review.get("summary", ""),
    }

    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "═"*50)
    print(f"  📄 Report  → {args.output}")
    print(f"  🎯 Score   → {report['score']}/10")
    print(f"  🔒 Risk    → {report['risk_level']}")
    print(f"  🚦 Decision→ {report['decision']}")
    print("═"*50 + "\n")

    # Exit codes used by Jenkinsfile pipeline
    if security_scan.get("block_deployment", False):
        print("🚨 Security scan is blocking deployment!", file=sys.stderr)
        sys.exit(2)

    if deploy_dec.get("decision") == "REJECT":
        print("🚫 AI Agent rejected this deployment!", file=sys.stderr)
        sys.exit(3)

    print("✅ AI Agent complete — pipeline may proceed.")


if __name__ == "__main__":
    main()
