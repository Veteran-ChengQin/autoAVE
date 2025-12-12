co#!/bin/bash
# 批量运行实验的脚本

# 设置API密钥（如果使用API模式）
# export DASHSCOPE_API_KEY="your-api-key-here"

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Attribute AKS Experiments Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查API密钥（对于API实验）
check_api_key() {
    if [ -z "$DASHSCOPE_API_KEY" ]; then
        echo -e "${YELLOW}Warning: DASHSCOPE_API_KEY not set. API experiments will fail.${NC}"
        echo -e "${YELLOW}Set it with: export DASHSCOPE_API_KEY='your-key'${NC}"
        echo ""
        return 1
    else
        echo -e "${GREEN}✓ API key found${NC}"
        return 0
    fi
}

# 运行单个实验
run_experiment() {
    local config_file=$1
    local exp_name=$(basename "$config_file" .yaml)
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running: $exp_name${NC}"
    echo -e "${GREEN}Config: $config_file${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    python main.py --config "$config_file"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $exp_name completed successfully${NC}"
    else
        echo -e "${RED}✗ $exp_name failed${NC}"
        return 1
    fi
    echo ""
}

# 主菜单
show_menu() {
    echo "Select experiments to run:"
    echo "  1) Exp1: Local model + Sampled frames"
    echo "  2) Exp2: API + Sampled frames"
    echo "  3) Exp3: API + All frames"
    echo "  4) Exp4: API + Video URL"
    echo "  5) Run all experiments"
    echo "  6) Run all API experiments (2-4)"
    echo "  q) Quit"
    echo ""
}

# 主循环
main() {
    while true; do
        show_menu
        read -p "Enter your choice: " choice
        
        case $choice in
            1)
                run_experiment "exp_demo/exp1_local_sampled_frames.yaml"
                ;;
            2)
                check_api_key && run_experiment "exp_demo/exp2_api_sampled_frames.yaml"
                ;;
            3)
                check_api_key && run_experiment "exp_demo/exp3_api_all_frames.yaml"
                ;;
            4)
                check_api_key && run_experiment "exp_demo/exp4_api_video_url.yaml"
                ;;
            5)
                echo -e "${YELLOW}Running all experiments...${NC}"
                run_experiment "exp_demo/exp1_local_sampled_frames.yaml"
                if check_api_key; then
                    run_experiment "exp_demo/exp2_api_sampled_frames.yaml"
                    run_experiment "exp_demo/exp3_api_all_frames.yaml"
                    run_experiment "exp_demo/exp4_api_video_url.yaml"
                fi
                ;;
            6)
                echo -e "${YELLOW}Running all API experiments...${NC}"
                if check_api_key; then
                    run_experiment "exp_demo/exp2_api_sampled_frames.yaml"
                    run_experiment "exp_demo/exp3_api_all_frames.yaml"
                    run_experiment "exp_demo/exp4_api_video_url.yaml"
                fi
                ;;
            q|Q)
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please try again.${NC}"
                echo ""
                ;;
        esac
    done
}

# 如果有命令行参数，直接运行对应实验
if [ $# -gt 0 ]; then
    case $1 in
        1|exp1)
            run_experiment "exp_demo/exp1_local_sampled_frames.yaml"
            ;;
        2|exp2)
            check_api_key && run_experiment "exp_demo/exp2_api_sampled_frames.yaml"
            ;;
        3|exp3)
            check_api_key && run_experiment "exp_demo/exp3_api_all_frames.yaml"
            ;;
        4|exp4)
            check_api_key && run_experiment "exp_demo/exp4_api_video_url.yaml"
            ;;
        all)
            run_experiment "exp_demo/exp1_local_sampled_frames.yaml"
            if check_api_key; then
                run_experiment "exp_demo/exp2_api_sampled_frames.yaml"
                run_experiment "exp_demo/exp3_api_all_frames.yaml"
                run_experiment "exp_demo/exp4_api_video_url.yaml"
            fi
            ;;
        *)
            echo -e "${RED}Unknown experiment: $1${NC}"
            echo "Usage: $0 [1|2|3|4|all]"
            exit 1
            ;;
    esac
else
    # 交互式菜单
    main
fi
