# Embed
Embedding is all your need.

# Further work

## Video
1. The heatmap could be easily exponental averaged between frames for
   better tackling with occlusion.

## Lidar
1. Lidar perception can also benefited from this architecture which make
   the appresiating `mid-fusion` scheme easy and efficient.

## 3D human pose/differetiable rendering
1. Make the expansion of 3D human pose and differetiable rendering from
   the single object to multi-object easily.

## Mul-model
1. Unify the embedding expression between image and texture to ease and improve
   the multi-model works such as image caption and image retrial and
   multi-model comprehension.

## 3D dynamic scene generation
1. Learning to extract an additional positional free embedding which we call it
   semantic/texture/apperance embedding.
2. Make the scene and appearence lantent embedding code more meaningful
   and releaf the buddern of the network expression and learning for `GIRAFFE`.

## Panoptic
1. Faster than `RealTimePan` on Cityscapes and Mapli Dataset with network
   architecture adaptation.
2. More accurate than current `PanopticFCN` by unify the training and
   inference embedding expression with affilant loss.
3. More accurate than current `PanopticFCN` by making the embedding
   contain more accurate positional information with affilant loss.
