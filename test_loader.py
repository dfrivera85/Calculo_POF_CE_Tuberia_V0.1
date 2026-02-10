
import data_loader
import os

try:
    print("Loading data from folder...")
    dfs = data_loader.load_data_from_folder(os.getcwd())
    print(f"Loaded {len(dfs)} dataframes")
    
    anomalias = dfs.get('anomalias')
    print(f"Anomalias dataframe shape: {anomalias.shape}")
    print(f"Anomalias columns: {anomalias.columns}")
    
    print("Creating master dataframe...")
    master_df = data_loader.create_master_dataframe(dfs)
    print("Master dataframe created successfully")
    print(f"Master dataframe shape: {master_df.shape}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
