#!/usr/bin/env python3
"""Utility class to find faces and facial features from image with mtcnn.
Inspired from example.py of MTCNN https://github.com/ipazc/mtcnn
"""
__author__ = "Jeff Zhu <zjxy63@gmail.com.ai>"


import argparse
import datetime
import json
import os
import sys

import cv2
from mtcnn.mtcnn import MTCNN

import numpy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.datasets import folder

# from tools.training.coco import dataset_creator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="XnorAI face finder script.")
    # Add the the rest of arguments
    parser.add_argument(
        "--verbose", action="store_true", help="Increase the output verbosity."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input image directory or a file.",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        help="Path to the output image directory with bounding boxes "
        "for detected faces and features.",
    )
    parser.add_argument(
        "--min_size",
        default=[128, 128],
        type=int,
        nargs="+",
        help="Face minimum size to be included.",
    )
    parser.add_argument(
        "--desired_features",
        default=["left_eye", "right_eye", "face"],
        nargs="+",
        required=False,
        help="Desired features to search for in the images. List of "
        "'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_left'",
    )
    parser.add_argument(
        "--feature_output_dir",
        required=False,
        help="Path to the output image directory for cropped facial features.",
    )
    parser.add_argument(
        "--found_output_file",
        required=False,
        help="Path to the output image file with name of images that found with"
        "desired featurs.",
    )
    parser.add_argument(
        "--not_found_output_file",
        required=False,
        help="Path to the output image file with name of images that not found "
        "with desired featurs.",
    )
    parser.add_argument(
        "--flat_categories",
        default=True,
        action="store_true",
        dest="flat_categories",
        help="Whether to use the one level flat categories (no parents).",
    )
    parser.add_argument(
        "--coco_json",
        required=False,
        help="Path to the coco dataset as a coco json file with annotations "
        "for the faces and facial features.",
    )
    parser.add_argument(
        "--no_visualize_output",
        action="store_false",
        dest="visualize_output",
        help="Whether to visualize the annotated image on the display.",
    )

    return parser.parse_args(argv)


def load_pillow_image(imagepath):
    with open(imagepath, "rb") as fp:
        image = Image.open(fp).convert("RGB")
    return image


class FaceFinder:
    """Find faces and facial features with MTCNN
    """

    def __init__(self):
        self._detector = MTCNN()

    def find_faces(self, image_file=None, pillow_image=None):
        """ Returns a list of with all the bounding boxes of faces detected

        Arguments:
           image_file: image file name

        Returns:
            [list]: bounding boxes of faces and keypoints for facial features.
            [
              {'keypoints': 
                {'right_eye': (314, 115), 
                  'nose': (303, 130), 
                   'left_eye': (291, 117),
                   'mouth_right': (313, 142),
                   'mouth_left': (296, 143)
                   }, 
                'box': [278, 92, 48, 62], 'confidence': 0.9999450445175171
                }, 
                {'keypoints': 
                  {'right_eye': (339, 191), 
                    'nose': (341, 199),
                    'left_eye': (327, 194),
                    'mouth_right': (342, 213),
                    'mouth_left': (334, 215)
                    },
                'box': [307, 173, 36, 54], 'confidence': 0.868172287940979}
                ]
        """

        if image_file is not None and not os.path.exists(image_file):
            raise Exception("Image path not exist")
            return []

        if image_file is not None:
            try:
                # print("Reading cv2.imread")
                image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(
                    "CV2 imread failed, image damaged or format not supported"
                )
                return []
        elif pillow_image is not None:
            try:
                # print("Converting to cv2 image")
                image = numpy.array(pillow_image)
            except Exception as e:
                print("Converting to CV2 image failed, Pillow image error.")
                return []

        result = self._detector.detect_faces(image)
        return result


def main(argv=None):
    args = parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir
    min_size = args.min_size
    desired_features = args.desired_features
    feature_output_dir = args.feature_output_dir
    coco_json = args.coco_json
    visualize_output = args.visualize_output
    minified = False
    flat_categories = args.flat_categories
    found_output_file = args.found_output_file
    not_found_output_file = args.not_found_output_file

    # make minimum face image size as square if only single number provided
    if len(min_size) < 2:
        min_size.append(image_size[0])

    # feature width, height, h-offset and v-offset scale with face bounding box
    feature_scales = {
        "right_eye": (4 / 9, 7 / 19, 0, 0),
        "nose": (3 / 8, 3 / 7, 0, -1 / 15),
        "left_eye": (4 / 9, 7 / 19, 0, 0),
        "mouth_right": (5 / 9, 2 / 9, -1 / 5, 0),
        "mouth_left": (5 / 9, 2 / 9, 1 / 5, 0),
    }
    categories_super = [
        {"id": 1, "name": "right_eye", "parents": [6], "supercategory": "eye"},
        {"id": 2, "name": "nose", "supercategory": "nose"},
        {"id": 3, "name": "left_eye", "parents": [6], "supercategory": "eye"},
        {
            "id": 4,
            "name": "mouth_left",
            "parents": [7],
            "supercategory": "mouth",
        },
        {
            "id": 5,
            "name": "mouth_right",
            "parents": [7],
            "supercategory": "mouth",
        },
        {"id": 6, "name": "eye", "supercategory": "eye"},
        {"id": 7, "name": "mouth", "supercategory": "mouth"},
        {"id": 8, "name": "face", "supercategory": "face"},
    ]
    categories_flat = [
        {"id": 1, "name": "right_eye", "supercategory": "right_eye"},
        {"id": 2, "name": "nose", "supercategory": "nose"},
        {"id": 3, "name": "left_eye", "supercategory": "left_eye"},
        {"id": 4, "name": "mouth_left", "supercategory": "mouth_left"},
        {"id": 5, "name": "mouth_right", "supercategory": "mouth_right"},
        {"id": 6, "name": "eye", "supercategory": "eye"},
        {"id": 7, "name": "mouth", "supercategory": "mouth"},
        {"id": 8, "name": "face", "supercategory": "face"},
    ]

    categories = categories_super if not flat_categories else categories_flat
    categories_name_to_id = {
        key: value
        for key, value in [(cat["name"], cat["id"]) for cat in categories]
    }
    info_dict = {
        "date_created": str(datetime.datetime.now()),
        "description": "dataset converted by face_finder.py",
    }
    annotations = []
    images = []

    if os.path.isdir(input_dir):
        image_files = [
            os.path.join(input_dir, filename)
            for filename in os.listdir(input_dir)
            if folder.is_image_file(os.path.join(input_dir, filename))
        ]
    else:
        image_files = [input_dir]

    finder = FaceFinder()

    annotation_count = 1
    list_files_feature_found = []
    list_files_feature_not_found = []
    # Find faces and facial feature for each image file.
    for image_id, filepath in enumerate(sorted(image_files), start=1):
        time_start = datetime.datetime.now()

        path, filename = os.path.split(filepath)

        image = load_pillow_image(filepath)
        width, height = image.size
        images.append(
            {
                "file_name": filename,
                "height": height,
                "id": image_id,
                "width": width,
            }
        )

        result = finder.find_faces(pillow_image=image)
        # print(result)

        time_end = datetime.datetime.now()
        time_elapsed = time_end - time_start

        # elapsed time microsecond per image
        elapsed_us = (
            time_elapsed.seconds * 1000000 + time_elapsed.microseconds
        ) / 1000000

        print("Finding faces took %8.3fs for image %d %s" % (elapsed_us, image_id, filename))

        # Result is an array with all the bounding boxes detected.
        output_image = image.copy()
        image_draw = ImageDraw.Draw(output_image)
        found_desired_feature = False
        for face_count, a_face in enumerate(result, start=1):
            bounding_box = a_face["box"]
            (left, top, width, height) = bounding_box
            # skipping small faces
            if width < min_size[0] or height < min_size[1]:
                continue

            (left, top) = (max(0, left), max(0, top))
            if "face" in desired_features:
                annotations.append(
                    {
                        "area": width * height,
                        "bbox": [left, top, width, height],
                        "category_id": categories_name_to_id["face"],
                        "id": annotation_count,
                        "image_id": image_id,
                        "iscrowd": 0,
                    }
                )
                annotation_count += 1
                # Draw face bounding_box
                image_draw.rectangle(
                    [(left, top), (left + width, top + height)],
                    fill=None,
                    outline=(255, 0, 0),
                    width=2,
                )

                # Output cropped feature images to feature_output_dir
                if feature_output_dir is not None:
                    eye_image = image.crop(
                        (left, top, left + width, top + height)
                    )
                    out_filename, ext = os.path.splitext(filename)
                    out_filename += "_" + "face_" + str(face_count) + ext
                    out_path = os.path.join(feature_output_dir, out_filename)

                    with open(out_path, "wb") as fp:
                        eye_image.save(fp)

            keypoints = a_face["keypoints"]
            for feature, scale in feature_scales.items():
                # Draw facial feature box
                if feature not in desired_features:
                    continue
                found_desired_feature = True
                feature_width = int(round(width * scale[0]))
                feature_height = int(round(height * scale[1]))
                (x, y) = keypoints[feature]
                feature_left = (
                    x - int(round(feature_width / 2) - width * scale[2])
                )
                feature_left = max(0, feature_left)
                feature_top = (
                    y - int(round(feature_height / 2) - height * scale[3])
                )
                feature_top = max(0, feature_top)
                # insert annotation for the feature
                annotations.append(
                    {
                        "area": feature_width * feature_height,
                        "bbox": [
                            feature_left,
                            feature_top,
                            feature_width,
                            feature_height,
                        ],
                        "category_id": categories_name_to_id[feature],
                        "id": annotation_count,
                        "image_id": image_id,
                        "iscrowd": 0,
                    }
                )
                annotation_count += 1

                # draw feature on image
                image_draw.rectangle(
                    [
                        (feature_left, feature_top),
                        (
                            feature_left + feature_width,
                            feature_top + feature_height,
                        ),
                    ],
                    fill=None,
                    outline=(255, 0, 0),
                    width=2,
                )

                # Output cropped feature images to feature_output_dir
                if feature_output_dir is not None:
                    feature_image = image.crop(
                        (
                            feature_left,
                            feature_top,
                            feature_left + feature_width,
                            feature_top + feature_height,
                        )
                    )
                    out_filename, ext = os.path.splitext(filename)
                    out_filename += "_" + str(face_count) + "_" + feature + ext
                    out_path = os.path.join(feature_output_dir, out_filename)

                    with open(out_path, "wb") as fp:
                        feature_image.save(fp)

        if found_desired_feature:
            list_files_feature_found.append(filename)
        else:
            list_files_feature_not_found.append(filename)
            
        if output_dir is not None:
            outfilepath = os.path.join(output_dir, filename)
            with open(outfilepath, "wb") as fp:
                output_image.save(fp)

        if visualize_output:
            # print("visualize_output for ", filepath)
            plt.imshow(output_image)
            plt.ion()
            plt.show()
            keyin = input('Press Enter to continue or "c" Enter skip prompt...')
            if len(keyin) > 0 and (keyin[0] == "c" or keyin[0] == "C"):
                visualize_output = False

    if coco_json:
        # Make the coco metadata dictionary
        dataset_metadata = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "info": info_dict,
            "type": "instances",
        }
        with open(coco_json, "w") as file_ptr:
            indent = None if minified else 2
            json.dump(dataset_metadata, file_ptr, indent=indent, sort_keys=True)

    if found_output_file is not None:
        with open(found_output_file, "w") as fp:
            fp.writelines(
                "{}\n".format(name) for name in list_files_feature_found
            )
    if not_found_output_file is not None:
        with open(not_found_output_file, "w") as fp:
            fp.writelines(
                "{}\n".format(name) for name in list_files_feature_not_found
            )


if __name__ == "__main__":
    main()
