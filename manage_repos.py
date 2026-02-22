import os
import re
import subprocess
import shutil
import stat
import sys

# ================= Configuration =================
# Set this to True to fetch the latest updates to all downloaded repositories.
# Set to False to skip repositories that are already downloaded.
LATEST = True 

# Output directory where the repositories will be arranged
OUTPUT_DIR = "Organized_Repos"
# =================================================

def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal. Useful for Windows .git folders."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def get_repo_links(file_path):
    """Read the Excel or Text file to extract GitHub repository URLs."""
    links = []
    if file_path.endswith(('.xlsx', '.xls')):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    for cell in row:
                        if isinstance(cell, str) and 'github.com' in cell:
                            links.append(cell.strip())
        except ImportError:
            print("Error: The 'openpyxl' library is required to read Excel files.")
            print("Please install it by running: pip install openpyxl")
            sys.exit(1)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if 'github.com' in line:
                    links.append(line)
    else:
        print("Unsupported file format. Please provide an .xlsx, .xls, or .txt file.")
        sys.exit(1)
        
    # Returning a unique list of links
    return list(set(links))

def extract_info_from_readme(repo_path):
    """Parse the README file in the repository to extract register number and name."""
    readme_names = ["README.md", "readme.md", "README.txt", "readme.txt", "README"]
    content = ""
    
    for name in readme_names:
        path = os.path.join(repo_path, name)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                break
            except Exception:
                pass
                
    if not content:
        return "UNKNOWN_REG", "Unknown_Name"
        
    # Search for University Register Number (e.g., TCR24CS029)
    # Pattern designed for KTU and similar roll numbers: 3 letters, 2 digits, 2 letters, 1-3 digits
    reg_no_match = re.search(r'\b[A-Za-z]{3}\d{2}[A-Za-z]{2}\d{1,3}\b', content)
    reg_no = reg_no_match.group(0).upper() if reg_no_match else "UNKNOWN_REG"
    
    # Search for exactly the pattern Name: <Name> or Student: <Name>
    name_match = re.search(r'(?i)(?:name|student name|student|author)\s*[:=-]\s*([A-Za-z .]+)', content)
    
    name = "Unknown_Name"
    if name_match:
        extracted_name = name_match.group(1).strip()
        if extracted_name:
            name = extracted_name
            
    return reg_no, name

def get_remote_url(repo_path):
    """Get the origin remote URL to track existing repositories."""
    try:
        result = subprocess.run(['git', '-C', repo_path, 'config', '--get', 'remote.origin.url'], 
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def process_repositories(file_path):
    """Main execution block to process the links, download, and organize folders."""
    links = get_repo_links(file_path)
    if not links:
        print(f"No GitHub links found in {file_path}.")
        return
        
    print(f"Found {len(links)} unique repository links.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Map existing downloaded repositories to handle 'latest' updates
    existing_repos = {}
    for entry in os.listdir(OUTPUT_DIR):
        full_path = os.path.join(OUTPUT_DIR, entry)
        if os.path.isdir(full_path) and os.path.isdir(os.path.join(full_path, '.git')):
            remote_url = get_remote_url(full_path)
            if remote_url:
                norm_remote = remote_url.replace('.git', '').rstrip('/')
                existing_repos[norm_remote] = full_path

    for link in links:
        norm_link = link.replace('.git', '').rstrip('/')
        
        # Check if the repository has already been cloned
        matched_repo_path = None
        for remote, path in existing_repos.items():
            if norm_link.endswith(remote) or remote.endswith(norm_link):
                matched_repo_path = path
                break
                
        if matched_repo_path:
            if LATEST:
                print(f"\n[UPDATE] Fetching latest updates for: {matched_repo_path}")
                subprocess.run(['git', '-C', matched_repo_path, 'pull'], check=False)
            else:
                print(f"\n[SKIP] Repository already exists: {matched_repo_path}")
            continue
            
        print(f"\n[CLONE] Downloading repository: {link}")
        temp_clone_dir = os.path.join(OUTPUT_DIR, "temp_clone_dir_1a2b3c")
        
        # Ensure temp directory doesn't already exist from a previous failed run
        if os.path.exists(temp_clone_dir):
            shutil.rmtree(temp_clone_dir, onerror=remove_readonly)
            
        try:
            # Execute standard git clone command
            subprocess.run(['git', 'clone', link, temp_clone_dir], check=True, capture_output=True)
            
            # Read README and extract the required formatting information
            reg_no, name = extract_info_from_readme(temp_clone_dir)
            
            # Form final layout for target folder
            if reg_no == "UNKNOWN_REG" and name == "Unknown_Name":
                repo_name = norm_link.split('/')[-1]
                target_folder_name = f"Unknown - {repo_name}"
            else:
                target_folder_name = f"{reg_no} - {name}"
                
            # Sanitize the folder name to be safe for Windows file paths
            target_folder_name = re.sub(r'[\\/*?:"<>|]', "", target_folder_name).strip()
            target_path = os.path.join(OUTPUT_DIR, target_folder_name)
            
            # Incrementally duplicate folder name mapping if conflicts happen
            counter = 1
            original_target_path = target_path
            while os.path.exists(target_path):
                target_path = f"{original_target_path} ({counter})"
                counter += 1
                
            # Move out of temp into the structured arrangement target directory!
            os.rename(temp_clone_dir, target_path)
            print(f"        -> Arranged in directory: {target_path}")
            
            # Cache the newly cloned item to track if identical repositories pop up
            existing_repos[norm_link] = target_path
            
        except subprocess.CalledProcessError:
            print(f"        -> [ERROR] Failed to clone. Checking access or repository validity: {link}")
        except Exception as e:
            print(f"        -> [ERROR] An unexpected error occurred: {e}")
        finally:
            # Force clean up strictly tracking temp leftovers
            if os.path.exists(temp_clone_dir):
                shutil.rmtree(temp_clone_dir, onerror=remove_readonly)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manage_repos.py <repository_links.xlsx | links.txt>")
        print("Note: ensure 'pip install openpyxl' is satisfied to process Excel files.")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' does not exist.")
        sys.exit(1)
        
    process_repositories(input_file)
