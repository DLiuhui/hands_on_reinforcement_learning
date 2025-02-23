from cliff_walking_env import CliffWalkingEnv

def main():
    env = CliffWalkingEnv()
    print(len(env.P), env.nrow, env.ncol)

if __name__ == "__main__":
    main()
