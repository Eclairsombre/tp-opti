def parse_vrp_file(filepath):
    """
    Lit un fichier .vrp et retourne un dictionnaire contenant :
      - 'name'        : nom du problème
      - 'capacity'    : capacité max des véhicules
      - 'depot'       : dict {x, y, ready, due}
      - 'clients'     : liste de dicts {id, x, y, ready, due, demand, service}
    """
    data = {}
    clients = []
    depot = None
    mode = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("NAME:"):
                data["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("NB_CLIENTS:"):
                data["nb_clients"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("MAX_QUANTITY:"):
                data["capacity"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("DATA_DEPOTS"):
                mode = "depot"
            elif line.startswith("DATA_CLIENTS"):
                mode = "clients"
            elif mode == "depot" and not line.startswith("DATA_"):
                parts = line.split()
                if len(parts) >= 5:
                    depot = {
                        "id": 0,
                        "name": parts[0],
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "ready": float(parts[3]),
                        "due": float(parts[4]),
                        "demand": 0,
                        "service": 0,
                    }
            elif mode == "clients" and not line.startswith("DATA_"):
                parts = line.split()
                if len(parts) >= 7:
                    idx = len(clients) + 1
                    clients.append(
                        {
                            "id": idx,
                            "name": parts[0],
                            "x": float(parts[1]),
                            "y": float(parts[2]),
                            "ready": float(parts[3]),
                            "due": float(parts[4]),
                            "demand": float(parts[5]),
                            "service": float(parts[6]),
                        }
                    )

    data["depot"] = depot
    data["clients"] = clients
    return data
