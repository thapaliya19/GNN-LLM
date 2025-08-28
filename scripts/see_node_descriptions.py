import pandas as pd

df = pd.read_csv("/home/ldn24004/autoprompt/output/europe/gemma3-12b/tta_history_seed_1.csv")
# how many times each node was high-entropy?
counts = df.groupby("node_idx").iteration.nunique()
print(counts.sort_values(ascending=False))

# for a given node, show its evolution:
node = 317
print(df[df.node_idx == node][["iteration","old_desc","new_desc"]])
for _, row in df[df.node_idx == node].iterrows():
    print(f"\n\nIteration {row.iteration}")
    print(f"\tNode {node}")
    print("  \nOLD DESC:")
    print(f"    {row.old_desc}")
    print("  \n NEW DESC:")
    print(f"    {row.new_desc}\n")
