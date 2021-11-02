import matplotlib.pyplot as plt
import numpy as np


def draw_tree(tree, args, ax):
    radius = 2.5
    formula = tree.primitive
    center_x, center_y = draw_node(formula, args, ax)

    if tree.left is not None:
        args = {'pose': 'left', 'parent_x': center_x, 'parent_y': center_y}
        p1 = [center_x - radius * np.sin(np.pi/4), center_y - radius * np.sin(np.pi/4)]
        child_x = center_x - 5
        child_y = center_y - 5
        p2 = [child_x + radius * np.sin(np.pi/4), child_y + radius * np.sin(np.pi/4)]
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        plt.plot(x_values, y_values, color = 'k')
        draw_tree(tree.left, args, ax)
    if tree.right is not None:
        args = {'pose': 'right', 'parent_x': center_x, 'parent_y': center_y}
        p1 = [center_x + radius * np.sin(np.pi/4), center_y - radius * np.sin(np.pi/4)]
        child_x = center_x + 5
        child_y = center_y - 5
        p2 = [child_x - radius * np.sin(np.pi/4), child_y + radius * np.sin(np.pi/4)]
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        plt.plot(x_values, y_values, color = 'k')
        draw_tree(tree.right, args, ax)


def draw_node(formula, args, ax):
    pose = args['pose']
    parent_x = args['parent_x']
    parent_y = args['parent_y']
    radius = 2.5

    if pose == 'root':
        node_center_x = 5
        node_center_y = 10

    elif pose == 'left':
        node_center_x = parent_x - 5
        node_center_y = parent_y - 5
    else:
        node_center_x = parent_x + 5
        node_center_y = parent_y - 5
    circle = plt.Circle((node_center_x, node_center_y), radius, color='b', fill=False)
    label = ax.annotate(formula, xy=(node_center_x, node_center_y), fontsize=5, ha="center")
    ax.add_patch(circle)
    return node_center_x, node_center_y




def main():
    args = {'pose': 'root', 'parent_x': None, 'parent_y': None}
    draw_tree(tree, args)
    treefig = plt.figure()


if __name__ == '__main__':
    main()
