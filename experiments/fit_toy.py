from approximation_eml.data import make_dataset, target_square_first, train_val_split
from approximation_eml.model import EMLTree
from approximation_eml.train import train_model
from approximation_eml.export import export_tree, summarize_structure
from approximation_eml.utils import set_seed


def main():
    set_seed(0)
    x, y = make_dataset(target_square_first, n=512, input_dim=2)
    x_train, y_train, x_val, y_val = train_val_split(x, y, frac=0.8)
    model = EMLTree(input_dim=2, depth=2, use_gates=True)
    history = train_model(model, x_train, y_train, x_val=x_val, y_val=y_val, config={})
    print(history)
    print(summarize_structure(model))
    print(export_tree(model))


if __name__ == "__main__":
    main()
