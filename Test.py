from Hypso_CNMF import run

delta = 0.8
names = ["frohavet_2024-05-06_1017Z-l1b", "caspiansea2_2024-08-15T06-53-10Z-l1b", "virginiabeach_2024-09-23T15-16-34Z-l1b"]

for i in range(6):
    # Update "delta" in config.txt
    with open("config.txt", "r") as config_file:
        config_lines = config_file.readlines()
    
    # Modify lines with "delta" and "type"
    for idx, line in enumerate(config_lines):
        if line.startswith("delta ="):
            config_lines[idx] = f"delta = {delta}\n"
        elif line.startswith("type ="):
            config_lines[idx] = "type = PPA\n"
    
    # Write updated config back
    with open("config.txt", "w") as config_file:
        config_file.writelines(config_lines)
    
    # Run for all names with "type = PPA"
    for j in range(3):
        run(names[j])
    
    # Update "type" to "CNMF" in config.txt
    for idx, line in enumerate(config_lines):
        if line.startswith("type ="):
            config_lines[idx] = "type = CNMF\n"
    
    # Write updated config back
    with open("config.txt", "w") as config_file:
        config_file.writelines(config_lines)
    
    # Run for all names with "type = CNMF"
    for j in range(3):
        run(names[j])
    
    # Halve the delta for the next iteration
    delta /= 2
