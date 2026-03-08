import json

def network_data_to_json(df, output_path,
                         time_col="time_interval",
                         id1_col="id_1",
                         id2_col="id_2"):
    """
    Transforms a standard network dataset into a time-sliced network JSON with the acceptable format for NetworkRugs.
    IMPORTANT: this function only stores the essential columns to visualize NetworkRugs.

    Required columns:
        time_col, id1_col, id2_col

    All other columns are added as edge attributes.

    Parameters
    ----------
    df : pandas.DataFrame
    output_path : str or Path
    """


    result = {}

    # detect node attribute pairs automatically
    attr_cols_1 = [c for c in df.columns if c.endswith("_1") and c != id1_col]
    attr_cols_2 = [c for c in df.columns if c.endswith("_2") and c != id2_col]

    for t, df_t in df.groupby(time_col):

        nodes = {}

        for _, row in df_t.iterrows():

            id1 = int(row[id1_col])
            id2 = int(row[id2_col])

            # Node 1
            if id1 not in nodes:
                node_data = {"id": id1}

                for col in attr_cols_1:
                    attr_name = col.replace("_1", "")
                    node_data[attr_name] = row[col]

                nodes[id1] = node_data

            # Node 2
            if id2 not in nodes:
                node_data = {"id": id2}

                for col in attr_cols_2:
                    attr_name = col.replace("_2", "")
                    node_data[attr_name] = row[col]

                nodes[id2] = node_data

        nodes_list = list(nodes.values())

        links_list = []
        for _, row in df_t.iterrows():
            links_list.append({
                "source": int(row[id1_col]),
                "target": int(row[id2_col]),
                "weight": 1
            })

        result[f"t_{t}"] = {
            "nodes": nodes_list,
            "links": links_list
        }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)