with open('Training/train_greece.py', encoding='utf-8') as f:
    content = f.read()

part1 = content.split('print(\n    "HYPER-HEURISTIC WARM-UP"\n)')[0]
part2 = content.split('print(\n    "FINAL TEST EVALUATION"\n)')[1]

replacement = """
# LOAD PRETRAINED MODEL FOR EVALUATION
print("Loading Kaggle checkpoint best_spatiotemporal_model_greece.pth")
checkpoint = torch.load("best_spatiotemporal_model_greece.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# We need the graph components.
model.active_k = checkpoint['decay_constant']
model.active_threshold = checkpoint['similarity_threshold']
model.locked_graph = True
print(f"Loaded Graph k: {model.active_k}, Threshold: {model.active_threshold}")

print(
    "FINAL TEST EVALUATION"
)
"""

new_content = part1 + replacement + part2

with open('Training/evaluate_greece_only.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print("Created evaluate_greece_only.py")
