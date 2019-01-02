# basic usage of pyplot

**start use**

```shell
# install 2.1.2 version for python 2.7
pip install matplotlib==2.1.2
```

**import plt**

```
import matplotlib.pyplot as plt
```

**get a figure**

```python
# specify the figure size
fig = plt.figure(figsize=(7.00, 5.25), clear=True)
```

**get axes**

```python
# specify the axes location
axe_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
```

**add plots**

```python
# set x axis data
x_axis = range(1, 11, 1)

# set y axis data
y_axis = range(1, 11, 1)

# add plot, set marker,label,linewidth,color
axe_1.plot(x_axis,y_axis,marker=".",label=r"$y=x$", linewidth=0.5, color="gray")

# set x label
axe_1.set_xlabel("x")

# set x limit
axe_1.set_xlim(left=1, right=10)

# set y label
axe_1.set_ylabel("y")

# set y limit
axe_1.set_ylim(bottom=1, top=10)

# set title
axe_1.set_title("y - x")

# show legend
axe_1.legend()

# show grid
axe_1.grid()

# show fig
fig.show()

# save fig with svg format
fig.savefig(fname="./test.svg", format="svg")

```

