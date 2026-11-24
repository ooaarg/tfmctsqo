"""SQL values, result decoding, and subprocess handling."""

import json
import math
import subprocess


def literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("SQL settings must be finite")
        return str(value)
    if not isinstance(value, str):
        raise ValueError(f"Expected a scalar setting, got {value!r}")
    return "'" + value.replace("'", "''") + "'"


def documents(text):
    decoder = json.JSONDecoder()
    result = []
    while text.strip():
        text = text.lstrip()
        value, end = decoder.raw_decode(text)
        result.append(value)
        text = text[end:]
    return result


def execute(command, script, timeout, output):
    output.with_suffix(".sql").write_text(script)
    try:
        process = subprocess.run(
            command, input=script, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        output.with_suffix(".stderr").write_text("Process timeout\n")
        raise TimeoutError("Process timeout") from exc
    output.with_suffix(".stdout").write_text(process.stdout)
    output.with_suffix(".stderr").write_text(process.stderr)
    if process.returncode:
        message = process.stderr.strip() or f"Process exited with {process.returncode}"
        if "timeout" in message.lower() or "interrupt" in message.lower():
            raise TimeoutError(message)
        raise RuntimeError(message)
    return process.stdout


def validate_settings(settings, allowed):
    unknown = settings.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported settings: {', '.join(sorted(unknown))}")
    for value in settings.values():
        literal(value)
