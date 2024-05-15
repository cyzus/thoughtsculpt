# thoughtsculpt

## Setup
```pip install -r requirements.txt```

```pip install -e .```

## Instruction

```
from thoughtsculpt.model.simulator import MCTS, DFS, ToT, COT
from thoughtsculpt.model.improver import ContentImprover

model_name = "gpt-3.5-turbo" # model of your choice
model = load_model(model_name=model_name, temp=0.7)
content_improver = ContentImprover(model=model, evaluator=None, solver_class=MCTS)

original_outlines = [
"Meet Snowball, a curious pet rabbit living in a cozy home with a loving family. Despite being confined to a cage, Snowball dreams of exploring the vast outdoors.",
"One day, Snowball's cage accidentally gets left open, and he seizes the opportunity to venture into the unknown garden. However, he soon realizes the dangers lurking outside and must find his way back home before it's too late.",
"Through his wit and determination, Snowball navigates the challenges of the outside world, learns valuable lessons about bravery and survival, and ultimately reunites with his worried but relieved family, realizing that sometimes, home truly is the safest place to be."
]

new_outlines_lst = content_improver.improve(original_outlines, depth=2, continous=True)

```
