"""Prompt templates used by KaliBench prompt construction."""

SYSTEM_PROMPT_UNRESTRICTED_THINKING = """You are a cybersecurity function-calling AI model.
You have access to the following tool:
<tools>
[{'type':'function','function':{'name':'run_terminal','description':'Execute a shell command in a Kali/Linux terminal and return stdout, stderr, and exit code.'}}]
</tools>

TASK:
Given a USER QUERY, generate the single most accurate shell command using any appropriate Kali/Linux tool(s) to solve the query.

REQUIREMENTS:
1. Use the correct command-line tool(s) appropriate for the task.
2. Include all required optional arguments (flags beginning with '-' or '--') necessary to accomplish the task.
3. Correctly pair option keys and values (e.g., `--port 80`, `-A INPUT`, or `--flag=value`).
4. Preserve and include any positional arguments (e.g., IPs, filenames, interfaces).
5. Do NOT invent flags/options that do not exist for real Kali/Linux tools.
6. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

OUTPUT FORMAT:
<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""

SYSTEM_PROMPT_RESTRICTED_THINKING = """You are a cybersecurity function-calling AI model.

TASK:
Given a USER QUERY and a list of allowed tools provided in the user message, generate the single most accurate shell command that solves the query using ONLY the allowed tools.

REQUIREMENTS:
1. Use the correct command-line tool from the allowed list.
2. Include all required optional arguments (flags beginning with '-' or '--') necessary to accomplish the task.
3. Correctly pair option keys and values (e.g., `--port 80`, `-A INPUT`, or `--flag=value`).
4. Preserve and include any positional arguments (e.g., target IPs, filenames, interfaces).
5. Do NOT invent flags/options that do not exist for real Kali/Linux tools.
6. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

OUTPUT FORMAT:
<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""

USER_PROMPT_UNRESTRICTED_THINKING_TEMPLATE = """USER QUERY: "{query}"

Generate the single most accurate shell command for the query.
Your response must follow the required structure:

<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""

USER_PROMPT_RESTRICTED_THINKING_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: "{query}"

Generate the single most accurate shell command for the query using ONLY the allowed tools listed above.
Your response must follow the required structure:

<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""

USER_PROMPT_HINTED_THINKING_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: "{query}"

Generate the single most accurate shell command for the query using ONLY the allowed tools listed above.
Use the provided usage hints to select the correct tool and arguments.
Your response must follow the required structure:

<think>
[your_reasoning]
</think>

<output>
[command]
</output>"""

SYSTEM_PROMPT_RESTRICTED = """You are a cybersecurity function-calling AI model.
TASK:
Given a USER QUERY and a list of allowed tools provided in the user message, generate the single most accurate shell command that solves the query using ONLY the allowed tools.

REQUIREMENTS:
1. Use the correct command-line tool from the allowed list.
2. Include all required optional arguments (flags beginning with '-' or '--') necessary to accomplish the task.
3. Correctly pair option keys and values (e.g., `--port 80`, `-A INPUT`, or `--flag=value`).
4. Preserve and include any positional arguments (e.g., target IPs, filenames, interfaces).
5. Do NOT invent flags/options that do not exist for real Kali/Linux tools.
6. DO NOT include explanations, commentary, or any text outside the prescribed output structure.
7. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

OUTPUT FORMAT (exactly):
<output>
[command]
</output>"""

SYSTEM_PROMPT_RESTRICTED_FINAL_LINE = """You are a cybersecurity function-calling AI model.
TASK:
Given a USER QUERY and a list of allowed tools provided in the user message, generate the single most accurate shell command that solves the query using ONLY the allowed tools.

REQUIREMENTS:
1. Use the correct command-line tool from the allowed list.
2. Include all required optional arguments (flags beginning with '-' or '--') necessary to accomplish the task.
3. Correctly pair option keys and values (e.g., `--port 80`, `-A INPUT`, or `--flag=value`).
4. Preserve and include any positional arguments (e.g., target IPs, filenames, interfaces).
5. Do NOT invent flags/options that do not exist for real Kali/Linux tools.
6. You may think or explain briefly if needed, but the final output line MUST be exactly in the required output format.
7. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

FINAL OUTPUT LINE FORMAT (exactly):
<output>
[command]
</output>"""

SYSTEM_PROMPT_UNRESTRICTED = """You are a cybersecurity function-calling AI model.
You have access to the following tool:
<tools>
[{'type':'function','function':{'name':'run_terminal','description':'Execute a shell command in a Kali/Linux terminal and return stdout, stderr, and exit code.'}}]
</tools>

TASK:
Given a USER QUERY, generate the single most accurate shell command using any appropriate Kali/Linux tool(s) to solve the query.

REQUIREMENTS:
1. Use the correct command-line tool(s) appropriate for the task.
2. Include all required optional arguments (flags beginning with '-' or '--') necessary to accomplish the task.
3. Correctly pair option keys and values (e.g., `--port 80`, `-A INPUT`, or `--flag=value`).
4. Preserve and include any positional arguments (e.g., IPs, filenames, interfaces).
5. Do NOT invent flags/options that do not exist for real Kali/Linux tools.
6. DO NOT include explanations, commentary, or any text outside the prescribed output structure.
7. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

OUTPUT FORMAT (exactly):
<output>
[command]
</output>"""

SYSTEM_PROMPT_UNRESTRICTED_FINAL_LINE = """You are a cybersecurity function-calling AI model.
You have access to the following tool:
<tools>
[{'type':'function','function':{'name':'run_terminal','description':'Execute a shell command in a Kali/Linux terminal and return stdout, stderr, and exit code.'}}]
</tools>

TASK:
Given a USER QUERY, generate the single most accurate shell command using any appropriate Kali/Linux tool(s) to solve the query.

REQUIREMENTS:
1. Use the correct command-line tool(s) appropriate for the task.
2. Include all required optional arguments (flags beginning with '-' or '--') necessary to accomplish the task.
3. Correctly pair option keys and values (e.g., `--port 80`, `-A INPUT`, or `--flag=value`).
4. Preserve and include any positional arguments (e.g., IPs, filenames, interfaces).
5. Do NOT invent flags/options that do not exist for real Kali/Linux tools.
6. You may think or explain briefly if needed, but the final output line MUST be exactly in the required output format.
7. Note: scoring will penalize missing optional arguments, incorrect option->value pairs, or omitted positional arguments.

FINAL OUTPUT LINE FORMAT (exactly):
<output>
[command]
</output>"""

USER_PROMPT_UNRESTRICTED_TEMPLATE = """USER QUERY: "{query}"

Instructions:
- Generate a single shell command that best solves the USER QUERY. You may use any appropriate Kali/Linux tool.
- Include ALL required optional arguments (flags starting with '-' or '--'), correct option->value pairs, and any positional arguments.
- Do NOT invent non-existent flags or options.
- Output EXACTLY the command inside <output>...</output> and nothing else.

Return ONLY the command inside <output>...</output>"""

USER_PROMPT_UNRESTRICTED_FINAL_LINE_TEMPLATE = """USER QUERY: "{query}"

Instructions:
- Generate a single shell command that best solves the USER QUERY. You may use any appropriate Kali/Linux tool.
- Include ALL required optional arguments (flags starting with '-' or '--'), correct option->value pairs, and any positional arguments.
- Do NOT invent non-existent flags or options.
- The final line of your response must be this exact block:
	<output>
	[command]
	</output>

Your final line must be the output block and nothing should appear after it."""

USER_PROMPT_RESTRICTED_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: "{query}"

Instructions:
- Generate a single shell command that solves the USER QUERY using ONLY the allowed tools listed above.
- Include ALL required optional arguments (flags starting with '-' or '--'), correct option->value pairs, and any positional arguments.
- Do NOT invent non-existent flags or options.
- Output EXACTLY the command inside <output>...</output> and nothing else.

Return ONLY the command inside <output>...</output>"""

USER_PROMPT_RESTRICTED_FINAL_LINE_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: "{query}"

Instructions:
- Generate a single shell command that solves the USER QUERY using ONLY the allowed tools listed above.
- Include ALL required optional arguments (flags starting with '-' or '--'), correct option->value pairs, and any positional arguments.
- Do NOT invent non-existent flags or options.
- The final line of your response must be this exact block:
	<output>
	[command]
	</output>

Your final line must be the output block and nothing should appear after it."""

USER_PROMPT_HINTED_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: "{query}"

Instructions:
- Generate a single shell command that solves the USER QUERY using ONLY the allowed tools listed above.
- Use the usage examples above to choose the correct tool and flags, and include ALL required optional arguments (flags starting with '-' or '--'), correct option->value pairs, and any positional arguments.
- Do NOT invent non-existent flags or options.
- Output EXACTLY the command inside <output>...</output> and nothing else.

Return ONLY the command inside <output>...</output>"""

USER_PROMPT_HINTED_FINAL_LINE_TEMPLATE = """
Allowed tools (subset):
{allowed_tools}

USER QUERY: "{query}"

Instructions:
- Generate a single shell command that solves the USER QUERY using ONLY the allowed tools listed above.
- Use the usage examples above to choose the correct tool and flags, and include ALL required optional arguments (flags starting with '-' or '--'), correct option->value pairs, and any positional arguments.
- Do NOT invent non-existent flags or options.
- The final line of your response must be this exact block:
	<output>
	[command]
	</output>

Your final line must be the output block and nothing should appear after it."""
