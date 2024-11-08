import os

# Define project structure with templates and static folders inside app
structure = {
    "requirements.txt": "flask\n",
    "app": {
        "__init__.py": " ",
        "route.py": " ",
        "templates": {  # Moved templates inside app
            "index.html": "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n<title>Home</title>\n</head>\n<body>\n<h1>User Interface</h1>\n</body>\n</html>\n"
        },
        "static": {  # Moved static inside app
            "style.css": "/* Add your CSS styles here */\n"
        },
    },
    "research": {"test.py":" "},
    "run.py": " ",
    "config.py":"config files",
    "Dockerfile":" ",
    "init_setup.sh":" "
}

# Function to create directory structure and files
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):  # It's a directory
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)  # Recursively create subdirectories
        else:  # It's a file
            with open(path, "w") as f:
                f.write(content)

# Create the project structure
project_root = os.getcwd()
create_structure(project_root, structure)

print("Flask project structure created successfully.")
