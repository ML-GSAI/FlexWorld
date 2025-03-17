import os
os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


class TRELLIS():
    def __init__(self, device='cuda'):
        # Load a pipeline from a model folder or a Hugging Face model hub.
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.to(device)

    def __call__(self, input_image, mesh_path = None,return_type=['mesh', 'gaussian']):
        """
            return_type: ['mesh', 'gaussian', 'radiance_field']
        """
        # Load an image
        image = Image.open(input_image)

        # Run the pipeline
        outputs = self.pipeline.run(
            image,
            # Optional parameters
            preprocess_image=False,
            formats = return_type
        )
        # outputs['gaussian'][0].save_ply("sample_gs.ply")


        # GLB files can be extracted from the outputs
        # glb = postprocessing_utils.to_glb(
        #     outputs['gaussian'][0],
        #     outputs['mesh'][0],
        #     # Optional parameters
        #     simplify=0.95,          # Ratio of triangles to remove in the simplification process
        #     texture_size=1024,      # Size of the texture used for the GLB
        # )
        # if mesh_path is None:
        #     glb.export(mesh_path)
        return outputs['gaussian'][0]
