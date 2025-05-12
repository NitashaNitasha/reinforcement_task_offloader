import os
import csv

def log_agent_actions_to_csv(episode_data, epoch, output_dir, agent_ids, name="mappo"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(os.getcwd()+output_dir, f"{name}_actions.csv")

    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ['epoch'] + [f'action_{agent_id}' for agent_id in agent_ids*100]
            writer.writerow(header)

        actions = episode_data['actions']  # e.g., [0, 1, 1, 0, 1]
        writer.writerow([epoch] + actions)



