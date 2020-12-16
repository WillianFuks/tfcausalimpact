docker run -v $(pwd):/tfcausalimpact quay.io/pypa/manylinux1_x86_64 sh -c '''
yum install -y json-c-devel
cd /tfcausalimpact
for PY in /opt/python/*/bin/; do
    if [[ ($PY != *"27"*) || ($PY != *"39"*)]]; then
        "${PY}/pip" install -U pip
        "${PY}/pip" install -U setuptools wheel auditwheel
        "${PY}/python" setup.py sdist bdist_wheel
    fi
done
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat "manylinux2010_x86_64" -w dist/
    rm $whl
done
'''
