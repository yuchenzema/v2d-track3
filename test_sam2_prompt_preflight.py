import ast
import importlib.util
from pathlib import Path
import sys
import unittest

BASE=Path(__file__).parent/'video_to_data/reconstruction/modules'
spec=importlib.util.spec_from_file_location('v2d_test_prompt_datatypes',BASE/'v2d_common/datatypes.py')
types=importlib.util.module_from_spec(spec);sys.modules[spec.name]=types;spec.loader.exec_module(types)
source=BASE/'v2d_sam2/lib/video_to_masks.py';tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_validate_prompt')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
validate=namespace['_validate_prompt']


class Sam2PromptPreflightTests(unittest.TestCase):
    def test_reviewed_positive_and_negative_points_are_a_single_valid_prompt_type(self):
        prompt=types.Sam2Prompt.from_dict(dict(frame_index=0,object_id=1,points=[dict(x=20,y=30),dict(x=40,y=50)],point_labels=[1,0]))
        validate(prompt)

    def test_combining_box_and_points_still_fails_the_real_validator(self):
        prompt=types.Sam2Prompt.from_dict(dict(frame_index=0,object_id=1,points=[dict(x=20,y=30)],point_labels=[1],box=dict(x0=0,y0=0,x1=100,y1=100)))
        with self.assertRaisesRegex(ValueError,'exactly one'):validate(prompt)

    def test_prompt_validation_precedes_loading_the_predictor(self):
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='video_to_masks')
        calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        validation=min(n.lineno for n in calls if n.func.id=='_validate_prompt')
        predictor=min(n.lineno for n in calls if n.func.id=='_get_predictor')
        self.assertLess(validation,predictor)


if __name__=='__main__':unittest.main()
