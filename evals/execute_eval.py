import argparse
import subprocess
import time
# kein saver '447967' '448071' '448115' '448610'
"""
Probleme:
'48583621'
'48584330',
'48554659'
"""
# done
"""
    '449407',
    '450474',
    '450802',
    '451425',
    '452244',
    '452256'
"""
def main():
    models = [
        '48638932',
        '48638932',
        '48638932',
        '48638932',
        '48638932',
        '48638932',
        '48638932',
        '48638932',
    ]
     # '447967', keine saver datei
    # Iterate over each model and execute a command
    for model in models:
        print(f"Executing command for model: {model}")
        
        # Replace 'echo' with the actual command you want to execute for each model
        command = ['slurmify', '1080-lo', 'python', f'/home/giese/Documents/gecco/evals/render_gaussians.py', f'--model_name', f'{model}','--variant', 'gen']
        
        # Execute the command
        try:
            subprocess.run(command, check=True)
            print(f"Command executed successfully for model: {model}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing command for model: {model}: {e}")
        
        # Wait for 3 minutes
        print("Waiting for 90 sec...")
        time.sleep(10)  # 180 seconds is 3 minutes
# unconditional: 452261
if __name__ == "__main__":
    main()