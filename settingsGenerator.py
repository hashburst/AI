def generate_config(api_key, miner, algo, pool, port, account, subaccount, coin, num_miners):
    config_filename = f"{api_key}_config.sh"
    with open(config_filename, "w") as config_file:
        for i in range(num_miners):
            config_file.write(f"./{miner} -a {algo} -o stratum+tcp://{pool}:{port} "
                              f"-u {account}.{subaccount} -p {subaccount} c={coin} &\n")
    print(f"Configuration file {config_filename} generated.")
