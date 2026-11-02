"""Run official DROID-SLAM, handling an empty set of proposed loop edges."""
import inspect
import runpy
import textwrap

import factor_graph as graph

source = textwrap.dedent(inspect.getsource(graph.FactorGraph.add_proximity_factors))
line = 'ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)'
assert source.count(line) == 1
source = source.replace(line, 'if not es:\n        return\n    ' + line)
namespace = {}
exec(compile(source, '<empty-proximity-edge-guard>', 'exec'), graph.__dict__, namespace)
graph.FactorGraph.add_proximity_factors = namespace['add_proximity_factors']
original_update = graph.FactorGraph.update_lowmem
def update_nonempty(self, *args, **kwargs):
    if self.ii.numel() == 0:
        return
    return original_update(self, *args, **kwargs)
graph.FactorGraph.update_lowmem = update_nonempty
runpy.run_module('v2d.droid_slam.lib.video_to_slam', run_name='__main__')
