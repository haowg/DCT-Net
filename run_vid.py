import os
import argparse
import imageio
import numpy as np

from tqdm import tqdm
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def process(args):
    print(f"style: {args.style}")
    style_dict = {"anime": "", "illustration": "-sd-illustration", "design": "-sd-design"}
    style = style_dict.get(args.style, "-" + args.style)

    model_name = 'damo/cv_unet_person-image-cartoon' + style + '_compound-models'
    print('model_path: ', model_name)
    img_cartoon = pipeline(Tasks.image_portrait_stylization, model=model_name)

    reader = imageio.get_reader(args.video_path)
    fps = reader.get_meta_data()['fps']
    duration = reader.get_meta_data()['duration']
    total_frames = int(fps * duration)

    with imageio.get_writer(args.save_path, mode='I', fps=fps, codec='libx264') as writer:
        for _, img in tqdm(enumerate(reader, 1), total=total_frames):
            result = img_cartoon(img[..., ::-1])
            res = result[OutputKeys.OUTPUT_IMG]
            writer.append_data(res[..., ::-1].astype(np.uint8))

    print('finished!')
    print(f'result saved to {args.save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='input.mp4')
    parser.add_argument('--save_path', type=str, default='res.mp4')
    parser.add_argument('--style', type=str, default='anime')
    args = parser.parse_args()

    process(args)
