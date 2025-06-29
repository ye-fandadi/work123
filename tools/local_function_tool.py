class LocalFunctionTool:
    def sort_list(self, lst):
        try:
            sorted_list = sorted(lst)
            return f"排序后的列表是：{sorted_list}"
        except Exception as e:
            return f"排序失败，错误信息：{e}"

