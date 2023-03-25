# PyTorch Template
> A clear PyTorch template for swift model building.

## Features
+ [x] Well-organized project template out of the box.
+ [x] Automatically record the model version (by saving the git commit hash) for later reproduction.
+ [x] Automatically start TensorBoard for you.
+ [x] Use JSON file or command line arguments to specify arguments.
+ [x] The results of each experiment are properly stored.


## Steps
1. Modify the model structures `models/build.py`.
2. Update the loss functions used `solver/loss.py`.
3. Update the data loading process `data/dataset.py` & `data/loader.py`.
4. Add metrics that can measure your model's performance `metrics/eval.py`.
5. Update sampling functions & logging functions, so you can see the results with TensorBoard `solver/solver.py`!
6. Add a shell script that run your model `scripts/{exp_id}-model_key_config.sh`.
7. Start training, evaluating or inference by running the above script!

## Structures
```
+--- .gitignore
+--- archive (generated files & dataset)
|   +--- README.md
+--- bin (utility script)
|   +--- README.md
|   +--- template.py
+--- config.py (options)
+--- data (data fetching related)
|   +--- dataset.py
|   +--- fetcher.py
|   +--- loader.py
|   +--- README.md
+--- expr (experiment directory)
+--- main.py (everything start from here)
+--- metrics (metric used)
|   +--- eval.py
|   +--- fid.py
|   +--- README.md
+--- models (model architecture related)
|   +--- build.py (the wrapper for models)
|   +--- discriminator.py
|   +--- generator.py
|   +--- layers.py
|   +--- mapping_network.py
|   +--- README.md
+--- README.md
+--- requirements.txt
+--- scripts (training related shell scripts)
|   +--- train.sh
+--- solver (training related)
|   +--- loss.py
|   +--- misc.py
|   +--- solver.py
|   +--- utils.py
+--- utils (utility functions)
|   +--- checkpoint.py
|   +--- file.py
|   +--- image.py
|   +--- logger.py
|   +--- misc.py
|   +--- model.py
```

## Others
I referred [StarGAN v2's official implementation](https://github.com/clovaai/stargan-v2) when crafting this template, 
so don't be surprised if you find some code is similar.

BTW, if you want to deploy your model, you may want to check out [this template](https://github.com/songquanpeng/pytorch-deployment).
