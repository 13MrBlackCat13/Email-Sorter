import os
import subprocess

def create_folders():
    folders = [
        'Unsorted',
        'text/Games', 'text/Newsletters', 'text/Notifications', 'text/Receipts', 'text/Social', 'text/Spam',
        'Sorted/Games', 'Sorted/Newsletters', 'Sorted/Notifications', 'Sorted/Receipts', 'Sorted/Social', 'Sorted/Spam',
        'emails/Games', 'emails/Newsletters', 'emails/Notifications', 'emails/Receipts', 'emails/Social', 'emails/Spam'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

def install_requirements():
    try:
        subprocess.run(['pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {str(e)}")

def main():
    create_folders()
    install_requirements()

if __name__ == "__main__":
    main()
