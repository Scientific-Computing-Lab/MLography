import superannotate as sa


# From SA format to COCO panoptic format
sa.export_annotation("/home/matanr/MLography/Segmentation/unet/data/grains_metallography/SA", 
                     "/home/matanr/MLography/Segmentation/unet/data/grains_metallography/COCO", 
                     "COCO", 
                     "grains", 
                     "Vector",
                     "instance_segmentation")