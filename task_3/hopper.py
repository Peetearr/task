import mujoco
import numpy as np
import mujoco.viewer
from lxml import etree
l1 = 2.25

class Hopper():

  def __init__(self):
      self.k = 20
      self.l1 = 1
      self.l2 = 1
      self.XML_PATH = 'task/task_3/hopper.xml'
      self.init_param()
      self.model = mujoco.MjModel.from_xml_path(self.XML_PATH)
      self.data = mujoco.MjData(self.model)
  
  def init_param(self):
    xml_tree = etree.parse(self.XML_PATH)
    mjcf_container = xml_tree.getroot()
    mass = mjcf_container.find(".//body[@name=\"mass\"]")
    mass.attrib['pos'] = "0 0 "+str(self.l2+self.l1+0.25)
    l1_container = mjcf_container.find(".//body[@name=\"leg1\"]")
    l1_container.attrib['pos'] = "0 0 "+str(-self.l2)
    l2_container = mjcf_container.find(".//geom[@name=\"l2\"]")
    l2_container.attrib['fromto'] = "0 0 "+str(-self.l2)+" 0 0 0"
    l1_geom_container = mjcf_container.find(".//geom[@name=\"l1\"]")
    l1_geom_container.attrib['fromto'] = "0 0 "+str(-self.l1)+" 0 0 0"
    spring_container = mjcf_container.find(".//joint[@name=\"hinge_2\"]")
    spring_container.attrib['stiffness'] = str(self.k)
    spring_container.attrib['damping'] = str(self.k/20)
    with open(str(self.XML_PATH), "wb") as file:
      file.write(etree.tostring(xml_tree))
