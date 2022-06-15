# PyTorch Template
> A clear PyTorch template for swift model building.

## Features
+ [x] Well-organized project template out of the box.
+ [x] Automatically record the model version (by saving the git commit hash) for later reproduction.
+ [x] Use JSON file or command line arguments to specify arguments.
+ [x] The results of each experiment are properly stored.


## Structures
```
.
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
+--- main.py (everthing start from here)
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
so don't be surprised when you see some code is similar.