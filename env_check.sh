#!/bin/bash
echo "=== which python ===" 
which python 
echo "=== which pip ===" 
which pip
echo "=== PYTHONPATH ===" 
echo "${PYTHONPATH:-not set}"
echo "=== PIP_TARGET ===" 
echo "${PIP_TARGET:-not set}"
echo "=== pip config list ===" 
pip config list
echo "=== pip config debug ===" 
pip config debug
echo "=== env PYTHON vars ===" 
env | grep -E "PYTHON|PIP" || echo "none found"
echo "=== /etc/environment ===" 
cat /etc/environment 2>/dev/null || echo "not found"
echo "=== /etc/profile.d/*.sh ===" 
ls /etc/profile.d/ 2>/dev/null
echo "=== bashrc ===" 
cat /home/openvscode-server/.bashrc 2>/dev/null || echo "not found"
echo "=== bash_profile ===" 
cat /home/openvscode-server/.bash_profile 2>/dev/null || echo "not found"
echo "=== profile ===" 
cat /home/openvscode-server/.profile 2>/dev/null || echo "not found"
echo "=== root bashrc ===" 
cat /root/.bashrc 2>/dev/null || echo "not found"
echo "=== pip.conf locations ===" 
find / -name "pip.conf" 2>/dev/null | head -20
echo "=== pip.ini locations ===" 
find / -name "pip.ini" 2>/dev/null | head -20
echo "=== classiq-venv exists? ===" 
ls /classiq-venv/ 2>/dev/null | head -5 || echo "/classiq-venv not found"
ls /home/openvscode-server/classiq-venv/ 2>/dev/null | head -5 || echo "/home/openvscode-server/classiq-venv not found"
echo "=== DONE ==="