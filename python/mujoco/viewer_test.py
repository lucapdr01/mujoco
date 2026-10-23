# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for mujoco.viewer that do not open a window."""

import os

from absl import flags
from absl.testing import absltest
import mujoco
from mujoco import viewer

_XML = """
<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom size=".1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _FakeSimulate:
  """Records the calls which `viewer._reload` makes to a Simulate instance."""

  def __init__(self):
    self.load_error = ''
    self.loaded = []

  def load_message(self, path):
    del path

  def load_message_clear(self):
    pass

  def load(self, m, d, path, spec=None):
    self.loaded.append((m, d, path, spec))


class ViewerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Mark flags as parsed to avoid pytest errors about unparsed flags.
    # This is needed for `create_tempdir()` calls below.
    flags.FLAGS.mark_as_parsed()

  def test_launch_function_exists(self):
    self.assertIsNotNone(viewer.launch)

  def test_model_loader_returns_spec(self):
    spec = mujoco.MjSpec.from_string(_XML)
    model = spec.compile()
    m, d, path, s = viewer._model_loader(model, spec=spec)()  # pylint: disable=protected-access
    self.assertIs(m, model)
    self.assertIsInstance(d, mujoco.MjData)
    self.assertEqual(path, '')
    self.assertIs(s, spec)

  def test_model_loader_keeps_data(self):
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    m, d, path, s = viewer._model_loader(model, data)()  # pylint: disable=protected-access
    self.assertIs(m, model)
    self.assertIs(d, data)
    self.assertEqual(path, '')
    self.assertIsNone(s)

  def test_reload_passes_spec(self):
    spec = mujoco.MjSpec.from_string(_XML)
    model = spec.compile()
    data = mujoco.MjData(model)
    sim = _FakeSimulate()
    viewer._reload(sim, viewer._model_loader(model, data, spec))  # pylint: disable=protected-access
    self.assertEqual(sim.loaded, [(model, data, '', spec)])

  def test_reload_from_file_has_no_spec(self):
    path = os.path.join(self.create_tempdir().full_path, 'model.xml')
    with open(path, 'w') as f:
      f.write(_XML)
    sim = _FakeSimulate()
    viewer._reload(sim, viewer._file_loader(path))  # pylint: disable=protected-access
    self.assertLen(sim.loaded, 1)
    _, _, loaded_path, spec = sim.loaded[0]
    self.assertEqual(loaded_path, path)
    self.assertIsNone(spec)

  def test_reload_from_user_loader_has_no_spec(self):
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    sim = _FakeSimulate()
    viewer._reload(sim, lambda: (model, data))  # pylint: disable=protected-access
    self.assertEqual(sim.loaded, [(model, data, '', None)])

  def test_launch_spec_without_model_raises(self):
    spec = mujoco.MjSpec.from_string(_XML)
    with self.assertRaisesRegex(ValueError, 'mjSpec is specified'):
      viewer.launch(spec=spec)

  def test_launch_spec_with_loader_raises(self):
    spec = mujoco.MjSpec.from_string(_XML)
    model = spec.compile()
    with self.assertRaisesRegex(ValueError, 'spec and loader are both'):
      viewer.launch(loader=lambda: (model, mujoco.MjData(model)), spec=spec)

  def test_launch_spec_wrong_type_raises(self):
    model = mujoco.MjModel.from_xml_string(_XML)
    with self.assertRaisesRegex(ValueError, 'is not a mujoco.MjSpec'):
      viewer.launch(model, spec=_XML)


if __name__ == '__main__':
  absltest.main()
