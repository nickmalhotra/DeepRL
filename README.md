# DeepRL
This repository is kept to host deep RL codes
This initial code that I have kept enables you to write a simple RL for the famous Super Mario Bors game at Nintendo. 

REPOSITORY
The first file here is the Mario_RL.py. The file contains comments and description to train a reinforcement learning player
The __main__ method contains 4 calls 
#Main method
if __name__ == "__main__":
    #check_environment()
    stacked_env = make_stacked_environment()
    train_environment(stacked_env)
    #run_trained_environment(stacked_env)
    
 1. Use check_environment method only to check the environment only
 2. For training the game on your GPU/CPU, recommendatio is to keep make_stacked_environment and train environment both uncommented
 3. Once trained the saved model would go to saved model path which I have kept as saved_models\PPO_Mario.YOu can keep a simlar path or change on line 51.
