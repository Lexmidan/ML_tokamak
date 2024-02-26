import os
import stat

def change_permissions(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

if __name__ == "__main__":
    folder_path = "/compass/Shared/Users/bogdanov/gamma_spice_sim"
    change_permissions(folder_path)
    print("Permissions changed for all files in the folder.")