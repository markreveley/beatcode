---
title: "Adding Custom Tools to Letta Agents: From Chatbot to Action-Taker"
date: "2025-01-28"
description: "Learn how to extend Letta agents with custom tools and functions, enabling them to interact with APIs, databases, file systems, and the real world."
tags: ["Letta", "Tutorial", "Tools", "Function Calling", "Integration"]
---

# Adding Custom Tools to Letta Agents: From Chatbot to Action-Taker

So far, we've built Letta agents that can remember and converse. But what if your agent could **do things**? Send emails, query databases, control smart home devices, or call external APIs? That's where **tools** come in.

In this tutorial, we'll transform our agents from passive conversationalists into active assistants that can take real-world actions.

## What Are Tools in Letta?

Tools (also called functions) are Python functions that Letta agents can call during conversations. When an agent needs to perform an action, it:

1. Decides which tool to use
2. Determines the parameters
3. Calls the function
4. Receives the result
5. Continues the conversation with that information

Think of tools as giving your agent "hands" to interact with the world.

## Prerequisites

- Completed previous tutorials (or familiar with Letta basics)
- Python 3.10+
- Letta installed (`pip install letta`)

## Our First Tool: A Simple Calculator

Let's start with a basic example - a calculator tool.

### Step 1: Define the Tool Function

```python
from letta import create_client
from letta.schemas.tool import Tool

def calculator(operation: str, num1: float, num2: float) -> str:
    """
    Perform basic math operations.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        num1: First number
        num2: Second number

    Returns:
        The result of the calculation
    """
    operations = {
        "add": num1 + num2,
        "subtract": num1 - num2,
        "multiply": num1 * num2,
        "divide": num1 / num2 if num2 != 0 else "Cannot divide by zero"
    }

    result = operations.get(operation.lower(), "Invalid operation")
    return f"Result: {result}"
```

Key points:
- **Type hints are required**: Letta uses them to understand parameters
- **Docstring is important**: It helps the agent understand when to use the tool
- **Return a string**: Results should be string-serializable

### Step 2: Register the Tool

```python
client = create_client()

# Create a Tool object from our function
calc_tool = Tool.from_function(calculator)

# Register it with Letta
client.create_tool(calc_tool)

print("✓ Calculator tool registered!")
```

### Step 3: Create an Agent with the Tool

```python
# Create an agent that can use the calculator
agent = client.create_agent(
    name="MathAssistant",
    tools=["calculator"],  # List of tool names
    persona="You are a math assistant. When users ask for calculations, use the calculator tool to provide accurate results."
)

# Test it out
response = client.send_message(
    agent_id=agent.id,
    message="What's 157 multiplied by 89?",
    role="user"
)

# The agent will automatically use the calculator tool!
```

## Practical Example: Weather Tool

Let's build something more useful - a weather lookup tool.

### Step 1: Install Dependencies

```bash
pip install requests
```

### Step 2: Create the Weather Tool

```python
import requests
from typing import Optional

def get_weather(city: str, country_code: Optional[str] = "US") -> str:
    """
    Get current weather for a city.

    Args:
        city: Name of the city
        country_code: Two-letter country code (default: US)

    Returns:
        Weather information including temperature and description
    """
    # Using a free weather API (you'll need an API key)
    API_KEY = "your_openweather_api_key"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country_code}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code == 200:
            temp = data['main']['temp']
            description = data['weather'][0]['description']
            humidity = data['main']['humidity']

            return f"Weather in {city}: {temp}°C, {description}, Humidity: {humidity}%"
        else:
            return f"Could not fetch weather for {city}"

    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# Register the tool
weather_tool = Tool.from_function(get_weather)
client.create_tool(weather_tool)

# Create an agent
weather_agent = client.create_agent(
    name="WeatherBot",
    tools=["get_weather"],
    persona="You are a helpful weather assistant. When users ask about weather, use the get_weather tool to provide accurate, real-time information."
)
```

### Step 3: Test the Weather Agent

```python
response = client.send_message(
    agent_id=weather_agent.id,
    message="What's the weather like in Tokyo?",
    role="user"
)

# Agent calls get_weather("Tokyo", "JP") and responds with the result
```

## Advanced Example: File System Tools

Let's create tools for reading and writing files, making our agent a file manager.

```python
import os
from pathlib import Path
from typing import Optional

def read_file(filepath: str) -> str:
    """
    Read contents of a text file.

    Args:
        filepath: Path to the file to read

    Returns:
        File contents or error message
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File contents:\n{content}"
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(filepath: str, content: str, append: bool = False) -> str:
    """
    Write content to a text file.

    Args:
        filepath: Path to the file
        content: Content to write
        append: If True, append to file; if False, overwrite (default: False)

    Returns:
        Success or error message
    """
    try:
        mode = 'a' if append else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(content)
        action = "Appended to" if append else "Wrote to"
        return f"✓ {action} '{filepath}' successfully"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def list_directory(directory: str = ".") -> str:
    """
    List contents of a directory.

    Args:
        directory: Path to directory (default: current directory)

    Returns:
        List of files and directories
    """
    try:
        items = os.listdir(directory)
        files = [f for f in items if os.path.isfile(os.path.join(directory, f))]
        dirs = [d for d in items if os.path.isdir(os.path.join(directory, d))]

        result = f"Contents of '{directory}':\n"
        result += f"Directories: {', '.join(dirs) if dirs else 'None'}\n"
        result += f"Files: {', '.join(files) if files else 'None'}"
        return result
    except Exception as e:
        return f"Error listing directory: {str(e)}"

# Register all tools
for func in [read_file, write_file, list_directory]:
    tool = Tool.from_function(func)
    client.create_tool(tool)

# Create a file manager agent
file_agent = client.create_agent(
    name="FileManager",
    tools=["read_file", "write_file", "list_directory"],
    persona="""You are a file management assistant. You can:
    - Read file contents
    - Write or append to files
    - List directory contents

    Always confirm actions with the user before writing or deleting files.
    Use absolute paths when possible to avoid confusion.
    """
)
```

### Using the File Manager

```python
# Example conversation
messages = [
    "What files are in the current directory?",
    "Read the contents of README.md",
    "Create a new file called notes.txt with the content 'Meeting at 3pm'"
]

for msg in messages:
    response = client.send_message(
        agent_id=file_agent.id,
        message=msg,
        role="user"
    )
    print(f"\nUser: {msg}")
    print(f"Agent: {response.messages[-1].text}")
```

## Building a Database Tool

Here's a more advanced example - querying a SQLite database:

```python
import sqlite3
from typing import List, Optional

def query_database(query: str, database_path: str = "data.db") -> str:
    """
    Execute a SELECT query on a SQLite database.

    Args:
        query: SQL SELECT query to execute
        database_path: Path to SQLite database file

    Returns:
        Query results formatted as a string
    """
    # Security: Only allow SELECT queries
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for safety"

    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(query)

        # Get column names
        columns = [description[0] for description in cursor.description]

        # Get results
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Query returned no results"

        # Format results
        result = f"Found {len(rows)} result(s):\n\n"
        result += " | ".join(columns) + "\n"
        result += "-" * 50 + "\n"

        for row in rows[:10]:  # Limit to 10 rows
            result += " | ".join(str(val) for val in row) + "\n"

        if len(rows) > 10:
            result += f"\n... and {len(rows) - 10} more rows"

        return result

    except Exception as e:
        return f"Database error: {str(e)}"

# Register the tool
db_tool = Tool.from_function(query_database)
client.create_tool(db_tool)

# Create database analyst agent
db_agent = client.create_agent(
    name="DatabaseAnalyst",
    tools=["query_database"],
    persona="""You are a database analyst assistant. You can query SQLite databases
    using the query_database tool.

    When users ask questions about data:
    1. Formulate an appropriate SELECT query
    2. Execute it using the tool
    3. Explain the results in plain language

    Always use safe SQL practices and never modify data.
    """
)
```

## Tool Best Practices

### 1. Clear Docstrings

```python
# Good: Detailed and clear
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email to a recipient.

    Args:
        to: Email address of recipient
        subject: Email subject line
        body: Email body content

    Returns:
        Success or failure message
    """
    pass

# Bad: Unclear
def send_email(to, subject, body):
    """Sends email"""
    pass
```

### 2. Error Handling

Always handle errors gracefully:

```python
def api_call(endpoint: str) -> str:
    """Call an external API."""
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        return "Error: Request timed out"
    except requests.RequestException as e:
        return f"Error: {str(e)}"
```

### 3. Type Hints

Letta needs type hints to understand parameters:

```python
# Good: Type hints provided
def calculate_age(birth_year: int, current_year: int) -> str:
    return f"Age: {current_year - birth_year}"

# Bad: No type hints
def calculate_age(birth_year, current_year):
    return f"Age: {current_year - birth_year}"
```

### 4. Return Strings

Tools should return strings or JSON-serializable types:

```python
# Good: Returns string
def get_user_info(user_id: int) -> str:
    user = fetch_user(user_id)
    return f"Name: {user.name}, Email: {user.email}"

# Works: Returns dict (will be JSON serialized)
def get_user_info(user_id: int) -> dict:
    return {"name": "Alice", "email": "alice@example.com"}
```

## Complete Example: Personal Assistant Agent

Let's combine multiple tools into a comprehensive personal assistant:

```python
from datetime import datetime
import json

# Tool 1: Get current time
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in specified timezone."""
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} {timezone}"

# Tool 2: Create reminder (simplified)
def create_reminder(task: str, due_date: str) -> str:
    """
    Create a reminder for a task.

    Args:
        task: Description of the task
        due_date: Due date in YYYY-MM-DD format

    Returns:
        Confirmation message
    """
    # In production, this would save to a database
    return f"✓ Reminder created: '{task}' due on {due_date}"

# Tool 3: Search web (mock example)
def search_web(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: Search query

    Returns:
        Search results summary
    """
    # In production, integrate with a real search API
    return f"Search results for '{query}': [Mock results - integrate with real API]"

# Register all tools
for func in [get_current_time, create_reminder, search_web]:
    tool = Tool.from_function(func)
    client.create_tool(tool)

# Create personal assistant
assistant = client.create_agent(
    name="PersonalAssistant",
    tools=["get_current_time", "create_reminder", "search_web"],
    persona="""You are a helpful personal assistant. You can:
    - Tell the time
    - Create reminders for tasks
    - Search the web for information

    Be proactive and helpful. When users mention tasks, offer to create reminders.
    """
)

# Example conversation
conversation = [
    "What time is it?",
    "Remind me to call the dentist tomorrow",
    "Search for the best Italian restaurants near me"
]

for msg in conversation:
    response = client.send_message(
        agent_id=assistant.id,
        message=msg,
        role="user"
    )
```

## Testing Your Tools

Always test tools independently before giving them to agents:

```python
# Test the tool directly
result = calculator("multiply", 12, 8)
print(result)  # Should output: Result: 96

# Then test with agent
response = client.send_message(
    agent_id=agent.id,
    message="What's 12 times 8?",
    role="user"
)
```

## Debugging Tool Calls

View which tools the agent called:

```python
response = client.send_message(
    agent_id=agent.id,
    message="What's the weather in Paris?",
    role="user"
)

# Inspect messages for function calls
for msg in response.messages:
    if msg.message_type == "function_call":
        print(f"Tool called: {msg.function_call.name}")
        print(f"Arguments: {msg.function_call.arguments}")
    elif msg.message_type == "function_return":
        print(f"Tool result: {msg.function_return}")
```

## Security Considerations

When building tools, consider:

1. **Input validation**: Sanitize all inputs
2. **Rate limiting**: Prevent abuse of API-calling tools
3. **Permissions**: Don't give tools more access than needed
4. **Logging**: Track tool usage for debugging and security

Example with validation:

```python
def delete_file(filepath: str) -> str:
    """Delete a file (with safety checks)."""

    # Safety: Don't allow deleting system files
    dangerous_paths = ["/etc", "/sys", "/bin", "C:\\Windows"]
    if any(filepath.startswith(path) for path in dangerous_paths):
        return "Error: Cannot delete system files"

    # Safety: Confirm file exists and is not a directory
    if not os.path.isfile(filepath):
        return "Error: Not a valid file"

    try:
        os.remove(filepath)
        return f"✓ Deleted {filepath}"
    except Exception as e:
        return f"Error: {str(e)}"
```

## Next Steps

You've now learned how to extend Letta agents with custom tools! You can:
- Integrate with any Python library
- Call external APIs
- Interact with databases
- Control IoT devices
- And much more!

In the next tutorial, we'll explore **deploying Letta agents to production** - making your agents accessible via APIs and web interfaces.

## Resources

- [Letta Tools Documentation](https://docs.letta.ai/tools)
- [Tool Examples Repository](https://github.com/letta-ai/letta/tree/main/examples/tools)
- [Function Calling Best Practices](https://docs.letta.ai/best-practices/tools)

What tools will you build? Share your ideas in the comments!
