#!/bin/bash

# VSAG 构建并安装到虚拟环境
# 用途: 将修改后的 VSAG 构建为 Python 包并安装到当前虚拟环境

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}VSAG 构建与安装脚本${NC}"
echo -e "${GREEN}======================================${NC}"

# 1. 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ]; then
    echo -e "${RED}❌ 错误: 未检测到虚拟环境${NC}"
    echo -e "${YELLOW}请先激活虚拟环境:${NC}"
    echo -e "  source /home/mingqi/SAGE-DB-Bench/sage-db-bench/bin/activate"
    exit 1
fi

if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}✓ 检测到虚拟环境: $VIRTUAL_ENV${NC}"
elif [ -n "$CONDA_PREFIX" ]; then
    echo -e "${GREEN}✓ 检测到 Conda 环境: $CONDA_PREFIX${NC}"
fi

# 2. 检查 Python 版本
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓ Python 版本: $PYTHON_VERSION${NC}"

# 3. 进入 VSAG 目录
VSAG_DIR="/home/mingqi/SAGE-DB-Bench/algorithms_impl/vsag"
cd "$VSAG_DIR"
echo -e "${GREEN}✓ 工作目录: $VSAG_DIR${NC}"

# 4. 清理旧的构建文件（保留 build-release 以加快增量编译）
echo -e "\n${YELLOW}[1/5] 清理 Python 构建文件...${NC}"
# 不删除 build-release，保持增量编译
rm -rf python/build/*
rm -rf python/pyvsag.egg-info/*
rm -rf python/pyvsag/_version.py
rm -f python/pyvsag/_pyvsag*.so
echo -e "${GREEN}✓ 清理完成${NC}"

# 5. 配置 CMake (Release 模式, 启用 Python 绑定)
echo -e "\n${YELLOW}[2/5] 配置 CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_PYBINDS=ON \
      -DENABLE_TESTS=OFF \
      -DENABLE_EXAMPLES=OFF \
      -DENABLE_TOOLS=OFF \
      -DPython3_EXECUTABLE=$(which python3) \
      -B build-release \
      -G "Unix Makefiles" \
      -S .
echo -e "${GREEN}✓ CMake 配置完成${NC}"

# 6. 编译 C++ 库和 Python 绑定
echo -e "\n${YELLOW}[3/5] 编译 VSAG (使用 $(nproc) 个并行任务)...${NC}"
cmake --build build-release --parallel $(nproc)
echo -e "${GREEN}✓ 编译完成${NC}"

# 7. 准备 Python 包
echo -e "\n${YELLOW}[4/5] 准备 Python 包...${NC}"

# 复制编译好的 .so 文件到 python/pyvsag/
PYVSAG_SO=$(find build-release -name "_pyvsag*.so" | head -n 1)
if [ -z "$PYVSAG_SO" ]; then
    echo -e "${RED}❌ 错误: 未找到编译的 _pyvsag*.so 文件${NC}"
    exit 1
fi
echo -e "  找到: $PYVSAG_SO"
cp "$PYVSAG_SO" python/pyvsag/
echo -e "${GREEN}✓ Python 绑定已复制${NC}"

# 8. 安装到虚拟环境
echo -e "\n${YELLOW}[5/5] 安装 pyvsag 到虚拟环境...${NC}"
cd python
pip install -e . --force-reinstall --no-build-isolation
cd ..
echo -e "${GREEN}✓ 安装完成${NC}"

# 9. 验证安装
echo -e "\n${YELLOW}验证安装...${NC}"
python3 -c "import pyvsag; print(f'pyvsag 版本: {pyvsag.__version__}'); print('导入成功!')" || {
    echo -e "${RED}❌ 验证失败: 无法导入 pyvsag${NC}"
    exit 1
}
echo -e "${GREEN}✓ 验证成功${NC}"

echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}🎉 VSAG 构建与安装完成！${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "\n${YELLOW}使用方法:${NC}"
echo -e "  python3 -c 'import pyvsag; print(pyvsag.__version__)'"
echo -e "\n${YELLOW}测试示例:${NC}"
echo -e "  cd $VSAG_DIR"
echo -e "  python3 examples/python/example_hnsw.py"
