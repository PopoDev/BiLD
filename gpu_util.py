import xml.etree.ElementTree as ET
import json
from io import IOBase


def getUtilDetails(xml_io: IOBase):
    tree = ET.parse(xml_io)
    root = tree.getroot()
    results = dict()

    for gpu_item in root.findall("gpu"):
        curr_results = dict()
        gpu_id = gpu_item.get("id")

        memory_item = gpu_item.find("fb_memory_usage")

        curr_results["total_mem"] = memory_item.find("total").text
        curr_results["reserved_mem"] = memory_item.find("reserved").text
        curr_results["used_mem"] = memory_item.find("used").text
        curr_results["free_mem"] = memory_item.find("free").text

        utilization_item = gpu_item.find("utilization")

        curr_results["GPU Util"] = utilization_item.find("gpu_util").text
        curr_results["GPU Mem Util"] = utilization_item.find("memory_util").text

        results[f"GPU {gpu_id}"] = curr_results

if __name__ == "__main__":
    getUtilDetails("foo.xml")