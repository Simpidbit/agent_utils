import asyncio
import uuid
import json
import copy
from pathlib import Path
import shutil
import pickle

import traceback

from typing import Annotated, TypedDict
from typeguard import typechecked

import argparse

import fitz  # PyMuPDF
import simpidlog
from concurrent.futures import ProcessPoolExecutor, as_completed

from .llmapi import OnlineLLM, LLMResponseTypeError

_LOG_PREFIX = '@Simpidbit/agent_utils/pdf2markdown.py\n'
_TMPDIR = Path('/tmp')

parser = argparse.ArgumentParser(description = 'PDF -> markdown\nBy Simpidbit Isaiah <simpidbit@gmail.com>.')
parser.add_argument('cmd', choices = ['extract', 'print', 'retry', 'export'], help = '命令')
parser.add_argument('target', help = '目标文件路径，在 extract 下是 PDF 路径；在 print 或 export 下是 PKL 路径')
parser.add_argument('--output', help = '输出文件位置')
parser.add_argument('--workers', type = int, help = '将 PDF 文件提取为 PNG 图片的进程数', default = 20)
parser.add_argument('--dpi', type = int, help = '将 PDF 文件提取为 PNG 图片的 DPI', default = 144)
parser.add_argument('--logdir', help = '日志保存路径', default = str(_TMPDIR / 'agent_utils/pdf2markdown'))
parser.add_argument('--restore', help = 'PDFExtractor .pkl 恢复')
args = parser.parse_args()


class PDFPNG:
    def __init__(self) -> None:
        self.pngs: dict[int, bytes] = {}

    @typechecked
    def from_disk(self, pngdir: str | Path) -> None:
        pngdir = Path(pngdir)
        for pngfile in pngdir.iterdir():
            if pngfile.is_file() and pngfile.suffix == '.png':
                self.pngs[int(pngfile.stem)] = pngfile.read_bytes()

    @typechecked
    def from_pdf(self, pdfpath: str | Path) -> None:
        pngdir_tmp: Path = _TMPDIR / str(uuid.uuid4())
        _convert_pdf_to_pngs(pdfpath, pngdir_tmp, args.workers, args.dpi)
        self.from_disk(pngdir_tmp)
        shutil.rmtree(pngdir_tmp)

    @typechecked
    def __getitem__(self, id: int) -> bytes:
        return self.pngs[id]

    def __len__(self) -> int:
        return len(self.pngs)


class PDFContentEntry(TypedDict):
    id: list[int]
    title: str
    page: int

class PDFContent(TypedDict):
    left: int
    right: int
    content: list[PDFContentEntry]

@typechecked
def _check_pdf_exists(pdfpath: str | Path) -> None:
    '''检查 PDF 是否存在'''
    if isinstance(pdfpath, str):
        pdfpath = Path(pdfpath)

    if not pdfpath.exists():
        errmsg = _LOG_PREFIX + f'PDF file not exists: {str(pdfpath)}'
        simpidlog.error(errmsg)
        raise FileNotFoundError(errmsg)

@typechecked
def _get_pdf_page_count(pdfpath: str | Path) -> int:
    '''获取 PDF 页数'''
    _check_pdf_exists(pdfpath = pdfpath)

    doc = fitz.open(pdfpath)
    page_count: int = doc.page_count
    doc.close()

    return page_count

@typechecked
def _render_one_page(
    args: tuple[
        Annotated[str, 'pdfpath'], 
        Annotated[str, 'outputdir'],
        Annotated[int, 'left_page'],
        Annotated[int, 'right_page'],
        Annotated[int, 'dpi']
    ]
) -> None:
    '''渲染几页 PDF 为 PNG'''
    pdfpath, outputdir, left_page, right_page, dpi = args

    doc = fitz.open(pdfpath)

    try:
        for i in range(left_page - 1, right_page):
            page = doc.load_page(i)

            pix = page.get_pixmap(dpi = dpi, alpha = False)

            out_path = Path(outputdir) / f"{i + 1:05d}.png"
            pix.save(out_path)
    finally:
        doc.close()

@typechecked
def _convert_pdf_to_pngs(
    pdfpath: str | Path,
    outputdir: str | Path,
    workers: int,
    dpi: int
) -> None:
    '''多进程将 PDF 转为 PNG'''
    assert workers != 0
    _check_pdf_exists(pdfpath = pdfpath)

    page_count: int = _get_pdf_page_count(pdfpath = pdfpath)
    every_worker_task_count: int
    if page_count % workers == 0:
        every_worker_task_count: int = int(page_count / workers)
    else:
        every_worker_task_count: int = (page_count // workers) + 1

    worker_task_assignments: list[tuple[int, int]] = []
    crest_page_number: int = 1
    for i in range(workers):
        worker_task_assignments.append((crest_page_number, min(crest_page_number + every_worker_task_count - 1, page_count)))
        crest_page_number += every_worker_task_count

    tasks = [
        (str(pdfpath), str(outputdir), worker_task_assignments[i][0], worker_task_assignments[i][1], dpi)
        for i in range(len(worker_task_assignments))
    ]

    Path(outputdir).mkdir(parents = True, exist_ok = True)

    simpidlog.info(_LOG_PREFIX + f'开始将 PDF 转换为 PNG，共 {workers} 进程，{page_count} 页')
    with ProcessPoolExecutor(max_workers = workers) as executor:
        futures = [executor.submit(_render_one_page, task) for task in tasks]

        for future in as_completed(futures):
            future.result()
    simpidlog.info(_LOG_PREFIX + f'PDF -> PNG 已完成')

@typechecked
async def _is_this_page_content(
    pngs: PDFPNG,
    page_number: int
) -> bool:
    '''这一页是否是目录的一部分？'''
    llm = OnlineLLM()

    MAX_TRY: int = 5

    for i in range(MAX_TRY):
        llm_response: str = await llm.call_responses(
            system_prompt = '你只能说 yes 或者 no ，不能说任何一个其他的字、词或句子，只能说英文的 yes 或者 no ！！！',
            user_prompt = '这是一本书中的某一页，这一页是否是目录的一部分？'
                          '如果你判断这一页包含目录开头或目录结尾或目录中间部分的内容，请回答 yes ，'
                          '如果这一页并不包含目录的任何一部分，请回答 no 。',
            effort = 'medium',
            temperature = 0.0,
            file_bins = { f'{page_number:05d}.png': pngs[page_number] }
        )
        llm_response = llm_response.lower()

        if 'yes' in llm_response and 'no' not in llm_response:
            await llm.client.close()
            return True
        if 'no' in llm_response and 'yes' not in llm_response:
            await llm.client.close()
            return False
        else:
            simpidlog.debug(_LOG_PREFIX + f'_is_this_page_content(): llm_response = \"{llm_response}\"')

    await llm.client.close()
    errmsg = _LOG_PREFIX + f'_is_this_page_content(): 尝试 {MAX_TRY} 次后，模型仍输出非法结果'
    simpidlog.error(errmsg)
    raise RuntimeError(errmsg)

@typechecked
async def _search_for_content_range(
    pngs: PDFPNG
) -> tuple[int, int]:
    '''查找目录的开始和结束页面'''

    assert len(pngs) > 10

    simpidlog.info(_LOG_PREFIX + '_search_for_content_range(): Round 1.')
    results = await asyncio.gather(*[
        _is_this_page_content(pngs, i)
        for i in range(10, int(min(101, len(pngs) / 2)), 10)
    ])

    # 找开始和结束区间
    start_end_sections: list[int] = []
    last_page: int = 1
    last_value: bool = False
    for i in range(len(results)):
        page_number = (i + 1) * 10

        if last_value ^ results[i]:
            start_end_sections += [last_page, page_number]

        last_value = results[i]
        last_page = page_number

    start_left: int = start_end_sections[0]
    start_right: int = start_end_sections[1]
    end_left: int = start_end_sections[2]
    end_right: int = start_end_sections[3]

    simpidlog.info(_LOG_PREFIX + '_search_for_content_range(): Done.')
    simpidlog.info(_LOG_PREFIX + '_search_for_content_range(): Round 2.')
    results = await asyncio.gather(*(
        [
            _is_this_page_content(pngs, i)
            for i in range(start_left, start_right + 1)
        ]
            +
        [
            _is_this_page_content(pngs, i)
            for i in range(end_left, end_right + 1)
        ]
    ))

    start_section_results = results[: start_right - start_left + 1]
    end_section_results = results[start_right - start_left + 1: ]

    start_page: int = -1
    end_page: int = -1

    for i in range(len(start_section_results)):
        if start_section_results[i]:
            start_page = i + start_left
            break

    for i in range(1, len(end_section_results) + 1):
        if end_section_results[-i]:
            end_page = end_right - i + 1
            break

    simpidlog.info(_LOG_PREFIX + f'_search_for_content_range(): Done, [{start_page}, {end_page}].')

    return (start_page, end_page)

@typechecked
async def _extract_content(
    pngs: PDFPNG,
    left:int, 
    right:int
) -> PDFContent:
    '''目录信息提取器'''
    llm = OnlineLLM()

    MAX_TRY: int = 5

    for i in range(MAX_TRY):
        simpidlog.info(_LOG_PREFIX + f'_extract_content(): 开始提取目录信息，第 {i + 1} 次尝试')
        try:
            llm_response: str = await llm.call_responses(
                system_prompt = 
'''你是一个教材 PDF 目录提取器，某个教材 PDF 的每一页都会被渲染成 PNG 图像，其中包含目录的若干张 PNG 图像会按照页码顺序传给你，你需要从这些包含目录的 PNG 图像中提取出完整的正文目录信息。
你必须输出一段完整的 JSON 文本，输出的 JSON 格式有硬性要求，格式为：
{
    "content": [
        {
            "id": [<一级编号>, <二级编号>, ...],        // 假如此条目在目录中是 "2.1 范数"，则这一项的值就是 [2, 1]。如果目录中没有写 x.x.x 格式的编号，你要自行给每项条目按照顺序和从属关系安排编号，如果目录中的编号含字母，你要把字母改成数字，例如 a 改成 1, b 改成 2, 诸如此类。总之：必须保证每项目录条目都有分层级的数字编号。
            "title": "<章/节/小节等目录条目标题>",      // 这个标题只包含标题内容，要排除编号类的文字。比如 "第一章 最优化理论" 这个条目的标题就是 "最优化理论"，而非 "第一章 最优化理论"。
            "page": <目录中写明的对应页码>              // 如目录中未写明此条目对应的页码，则这个值置为 -1 ，但不能省略这一项。
        },
        {
            "id": [<一级编号>, <二级编号>, ...],        // 假如此条目在目录中是 "2.1 范数"，则这一项的值就是 [2, 1]。如果目录中没有写 x.x.x 格式的编号，你要自行给每项条目按照顺序和从属关系安排编号，如果目录中的编号含字母，你要把字母改成数字，例如 a 改成 1, b 改成 2, 诸如此类。总之：必须保证每项目录条目都有分层级的数字编号。
            "title": "<章/节/小节等目录条目标题>",      // 这个标题只包含标题内容，要排除编号类的文字。比如 "第一章 最优化理论" 这个条目的标题就是 "最优化理论"，而非 "第一章 最优化理论"。
            "page": <目录中写明的对应页码>              // 如目录中未写明此条目对应的页码，则这个值置为 -1 ，但不能省略这一项。
        },
        ......
    ]
}

注意：你只需要提取目录中的正文信息，请你直接忽略前言、序言、附录等非正文信息！

下面是一些输出示例，供你理解格式。

例1：
{
    "content": [
        {
            "id": [1],
            "title": "最优化简介",
            "page": 1
        },
        {
            "id": [1, 1],
            "title": "最优化问题概括",
            "page": 1
        },
        {
            "id": [1, 1, 1],
            "title": "最优化问题的一般形式",
            "page": 1
        },
        {
            "id": [1, 1, 2],
            "title": "最优化问题的类型与应用背景",
            "page": 2
        },
        {
            "id": [1, 2],
            "title": "实例：稀疏优化",
            "page": 3
        },
        {
            "id": [1, 3],
            "title": "实例：低秩矩阵恢复",
            "page": 7
        },
        {
            "id": [1, 4],
            "title": "实例：深度学习",
            "page": 8
        },
        {
            "id": [1, 4, 1],
            "title": "多层感知机",
            "page": 8
        },
        {
            "id": [1, 4, 2],
            "title": "卷积神经网络",
            "page": 10
        },
        {
            "id": [1, 5],
            "title": "最优化的基本概念",
            "page": 12
        },
        ......
    ]
}

例2：
{
    "content": [
        {
            "id": [1],
            "title": "神经网络的复习",
            "page": -1
        },
        {
            "id": [1, 1],
            "title": "数学和 Python 的复习",
            "page": 1
        },
        {
            "id": [1, 1, 1],
            "title": "向量和矩阵",
            "page": 1
        },
        ......
    ]
}

注意：你只能输出一整段完整的 JSON 文本，你的输出应该可以直接被 JSON 解析器解析成 JSON 对象。
不要有任何在 JSON 文本之外的回复、讨论或其他自然语言。''',
                user_prompt = '若干张目录图片已经上传给你，请你按照系统要求提取目录信息。',
                temperature = 0.0,
                effort = 'high',
                file_bins = {
                    f'{i:05d}.png': pngs[i]
                    for i in range(left, right + 1)
                }
            )
            simpidlog.info(_LOG_PREFIX + f'_extract_content(): 模型成功返回目录信息')
        except Exception as exc:
            traceback.print_exc()
            simpidlog.info(_LOG_PREFIX + f'_extract_content(): 模型调用失败')
            await llm.client.close()
            llm = OnlineLLM()
            continue

        await llm.client.close()

        try:
            res_dict: dict = llm.parse_json(llm_response)
        except:
            simpidlog.info(_LOG_PREFIX + f'_extract_content(): 模型输出目录信息 JSON 解析失败')
            continue

        simpidlog.info(_LOG_PREFIX + f'_extract_content(): 目录解析完毕')
        return {
            'content': res_dict['content'],
            'left': left,
            'right': right
        }

    await llm.client.close()
    errmsg = _LOG_PREFIX + f'_extract_content(): 尝试 {MAX_TRY} 次后，模型仍无法输出合法结果'
    simpidlog.error(errmsg)
    raise RuntimeError(errmsg)



'''
[
    {
        'id': 1,
        'title': ...,
        'text': ...,
        'childs': [
            {
                'id': 1,
                'title': ...,
                'text': ...,
                'childs': []
            },
            {
                'id': 1,
                'title': ...,
                'text': ...,
                'childs': [
                    {...},
                    ...
                ]
            }
        ]
    },
    ...
]
'''

class StructuredPDF:
    def __init__(self) -> None:
        self.data: list[dict] = []

    @typechecked
    def is_index_exists(self, index: list[int]) -> bool:
        curlist: list[dict] = self.data

        for id in index:
            found_tmp = False
            for each in curlist:
                if each['id'] == id:
                    curlist = each['childs']
                    found_tmp = True
                    break
            if not found_tmp:
                return False
        return True

    @typechecked
    def __getitem__(self, index: list[int]) -> dict:
        curlist: list[dict] = self.data
        curdict: dict = {}

        for id in index:
            found_tmp = False
            for each in curlist:
                if each['id'] == id:
                    curlist = each['childs']
                    curdict = each
                    found_tmp = True
                    break
            if not found_tmp:
                curlist.append({
                    'id': id,
                    'childs': []
                })
                curdict = curlist[-1]
                curlist = curlist[-1]['childs']
        return curdict

    @typechecked
    def set(
        self, 
        ids: list[int],
        *,
        title: str | None = None,
        page: int | None = None,
        text: str | None = None
    ) -> None:
        curdict = self.__getitem__(ids)

        if title:
            curdict['title'] = title

        if page:
            curdict['page'] = page

        if text:
            curdict['text'] = text

    @typechecked
    def next(
        self,
        ids: list[int]
    ) -> list[int]:
        curdict = self.__getitem__(ids)
        minid: int | None = None
        if curdict['childs']:
            for child_idx in range(len(curdict['childs'])):
                if minid is None:
                    minid = curdict['childs'][child_idx]['id']
                elif curdict['childs'][child_idx]['id'] < minid:
                    minid = curdict['childs'][child_idx]['id']
            assert minid is not None
            return copy.deepcopy(ids) + [minid, ]
        else:
            new_ids = copy.deepcopy(ids)
            new_ids[-1] += 1
            if self.is_index_exists(new_ids):
                return new_ids

            while new_ids[:-1]:
                new_ids = new_ids[:-1]
                new_ids[-1] += 1
                if self.is_index_exists(new_ids):
                    return new_ids
            return []

    def get_print_text(self) -> str:
        return json.dumps(self.data, indent = 4, ensure_ascii = False, default = str)

    @typechecked
    def from_json(self, jsonpath: str | Path) -> None:
        jsonpath = Path(jsonpath)
        with open(jsonpath, 'rt', encoding = 'utf-8') as f:
            self.data = json.loads(f.read())

@typechecked
async def _extract_one_section(
    pngs: PDFPNG,
    left_page: int,
    right_page: int,
    left_title: str,
    right_title: str | None
) -> str:
    '''让模型提取一节内容'''
    llm = OnlineLLM()

    MAX_TRY: int = 5

    if isinstance(right_title, str):
        user_prompt = f'请你提取标题 {left_title} 下的内容：也就是从标题 {left_title} 到标题 {right_title} 之间的内容。'
    else:
        user_prompt = f'请你提取标题 {left_title} 下的内容。'

    for i in range(MAX_TRY):
        try:
            llm_response = await llm.call_compatible(
                system_prompt = 
'''## 背景介绍
用户会上传若干张 PNG 图片给你，这若干张 PNG 图片其实是一个 PDF 文件当中的连续的几页渲染成 PNG 图片的结果。
这些 PNG 图片的上传顺序都是符合教材中的页码顺序的。
这个 PDF 文件通常是一本教材的电子版，拥有完整的目录和成体系的章节划分。
用户希望把这个 PDF 文件用 AI 一点一点把内容提取成 markdown 格式的纯文本，以方便大语言模型阅读。
但苦于大模型上下文长度有限，用户不可能一次性把整本书的内容都上传给大模型，再让大模型一次性提取出整本书——这是不可能的。
所以用户只能让 AI 一节一节地提取内容——这就是你现在需要做的工作。

## 工作描述与要求
用户上传给你的一组 PNG 图片当中，通常包含教材中的至少两个标题，还可能包含教材中的一些具体内容。
用户会告诉你需要提取从哪个标题到哪个标题之间的内容，用户确保你需要提取的那一部分已经在给你上传的图片当中了。
如果这组 PNG 图片当中有两个标题，通常来说，前一个标题就是用户需要你提取的那一小节的标题，后一个标题标志着前一个标题下面内容的结束，你不用把后一个标题也提取出来，你知道那标志着前一个标题下面的内容结束了就行。
这一前一后两个标题，可能是同级标题，比如都是三级标题；
也可能是不同级标题，比如前一个是一级标题，后一个是二级标题，类似这种情况下，可能的情况是，一级标题是一个章的标题，二级标题是这个章节下面某一节的标题，用户希望你把这个章的章头导言（也就是出现在章标题之后、第一个节标题之前的内容）提取出来。
当然，如果用户让你提取的这两个标题之间的确没有任何内容（的确有一些书没有章头导言，章标题下面直接就开始进入节标题），那你就输出一个 "无" 字就行。
上面我说的只是举了几个例子，真实的情况可能要更复杂，你需要领会我上面这些话传达出来的思想，根据情况随机应变。
总之，你的职责就是，忠实地提取两个标题之间的内容，和原文尽量保持一致，尽量做到一字不改。

## 补充要求
当你提取内容时，你可能遇到：
- **文本**：没什么好说的，忠实提取即可。
- **数学公式**：如果是行内公式，用 $ 包裹起来的 LaTeX 代码表达；如果是块级公式，用 $$ 包裹起来的 LaTeX 代码表达。
- **插图**：将插图用文本的形式详细描述出来，描述这张插图的形式、外观、要传达的核心信息、与上下文的联系和这张插图的作用。尽量用文字把这张插图的信息完整地表达出来，让人看了这段文字就好像已经看了这张插图。
- **表格、程序代码等 markdown 标准支持的内容类型**：使用对应的 markdown 语法表达。

## 格式要求
你需要直接输出你提取的结果，不要在你输出的提取结果外面再套一层 ```markdown 代码围栏，直接输出提取结果的 markdown 代码就行。
用户让你提取某一标题下的内容，这个标题不管是几级标题，在你输出的提取结果中统统作为开头的一级标题（而且不要把标题在原书中的编号写出来，只写标题本身，不带编号！），下面的标题层级以此递编。
''',
                user_prompt = user_prompt,
                temperature = 0.1,
                file_bins = {
                    f'{i:05d}.png': pngs[i]
                    for i in range(left_page, right_page + 1)
                },
                effort = 'xhigh'
            )
            simpidlog.debug(
                _LOG_PREFIX +
                    f'_extract_one_section(): 标题 \"{left_title}\"（ page: {left_page}~{right_page} ）下的内容提取成功，{len(llm_response)} 字\n{llm_response[:50]}...'
            )
            await llm.client.close()
            return llm_response
        except:
            simpidlog.warning(_LOG_PREFIX + f'_extract_one_section(): 标题 \"{left_title}\"（ page: {left_page}~{right_page} ）下的内容提取失败，重试第 {i + 1} 次...')
    
    await llm.client.close()
    simpidlog.warning(_LOG_PREFIX + f'_extract_one_section(): 标题 \"{left_title}\"（ page: {left_page}~{right_page} ）下的内容提取失败，重试 {MAX_TRY} 次后无果，已返回 \"UnknownError\"')
    return 'UnknownError'


class PDFExtractor:
    @typechecked
    def __init__(
        self,
        pdfcontent: PDFContent,
        pngs: PDFPNG,
        p1_page: int
    ) -> None:
        self.pdfcontent: PDFContent = pdfcontent
        self.pngs = pngs
        self.p1_page: int = p1_page
        self.data: StructuredPDF = StructuredPDF()

    def extract(self) -> StructuredPDF:
        self._from_content_to_data()
        asyncio.run(self._extract_text())
        return self.data

    def _from_content_to_data(self) -> None:
        for entry in self.pdfcontent['content']:
            ids: list[int] = entry['id']
            title: str = entry['title']
            page: int = entry['page']

            self.data.set(ids, title = title, page = page)

    async def _retry_entry(self, idss: list[list[int]]) -> None:
        tasks = []
        tasks_ids = []
        for ids in idss:
            next_ids: list[int] = self.data.next(ids)

            this_page: int = self.data[ids]['page']

            if next_ids:
                next_page: int = self.data[next_ids]['page']
                right_title = self.data[next_ids]['title']
            else:
                # TODO
                next_page: int = self.data[ids]['page'] + 10
                right_title = None


            tasks.append(_extract_one_section(
                pngs = self.pngs,
                left_page = this_page + self.p1_page - 1,
                right_page = next_page + self.p1_page - 1,
                left_title = self.data[ids]['title'],
                right_title = right_title
            ))
            tasks_ids.append(ids)

        simpidlog.info(_LOG_PREFIX + f'_extract_text(): 提取内容中...')
        results = await asyncio.gather(*tasks)
        simpidlog.info(_LOG_PREFIX + f'_extract_text(): 内容提取完毕！')

        assert len(results) == len(tasks) == len(tasks_ids)
        for i in range(len(results)):
            self.data[tasks_ids[i]]['text'] = results[i]


    async def _extract_text(self) -> None:
        tasks = []
        tasks_ids = []
        for entry in self.pdfcontent['content']:
            ids: list[int] = entry['id']
            next_ids: list[int] = self.data.next(ids)

            this_page: int = self.data[ids]['page']

            if next_ids:
                next_page: int = self.data[next_ids]['page']
                right_title = self.data[next_ids]['title']
            else:
                # TODO
                next_page: int = self.data[ids]['page'] + 10
                right_title = None


            tasks.append(_extract_one_section(
                pngs = self.pngs,
                left_page = this_page + self.p1_page - 1,
                right_page = next_page + self.p1_page - 1,
                left_title = self.data[ids]['title'],
                right_title = right_title
            ))
            tasks_ids.append(ids)

        simpidlog.info(_LOG_PREFIX + f'_extract_text(): 提取内容中...')
        results = await asyncio.gather(*tasks)
        simpidlog.info(_LOG_PREFIX + f'_extract_text(): 内容提取完毕！')

        assert len(results) == len(tasks) == len(tasks_ids)
        for i in range(len(results)):
            self.data[tasks_ids[i]]['text'] = results[i]

def extract_handler() -> None:
    pdfpath: Path = Path(args.target)
    outdir: Path = Path(args.output)

    pngs: PDFPNG = PDFPNG()
    pngs.from_pdf(pdfpath = pdfpath)

    if args.restore is None:
        left, right = asyncio.run(_search_for_content_range(pngs))
        pdfcontent: PDFContent = asyncio.run(_extract_content(pngs, left, right))
        with open(outdir / (pdfpath.stem + '.content.json'), 'wt', encoding = 'utf-8') as f:
            f.write(json.dumps(pdfcontent, indent = 4, ensure_ascii = False))
        simpidlog.info(_LOG_PREFIX + f'请查看：{str(outdir / (pdfpath.stem + '.content.json'))}')
        input('Continue?')

        pdfextractor = PDFExtractor(pdfcontent, pngs, int(input('p1_page: ')))
        with open(outdir / (pdfpath.stem + '.pkl'), 'wb') as f:
            pickle.dump(pdfextractor, f)
    else:
        with open(args.restore, 'rb') as f:
            pdfextractor = pickle.load(f)

    input('Begin extract?')
    with open(outdir / (pdfpath.stem + '.json'), 'wt', encoding = 'utf-8') as f:
        f.write(json.dumps(pdfextractor.extract().data, indent = 4, ensure_ascii = False))
    with open(outdir / (pdfpath.stem + '.pkl'), 'wb') as f:
        pickle.dump(pdfextractor, f)
    simpidlog.info('Done.')

def _print_content_of_pdf(pdf: StructuredPDF):
    def _layer_print(level: int, layer: list[dict]):
        for entry in layer:
            print(' ' * 4 * level + f'{entry['id']}: {entry['title']} ({len(entry['text'])})')
            if len(entry['childs']) > 0:
                _layer_print(level + 1, entry['childs'])
    _layer_print(0, pdf.data)


def print_handler():
    jsonpath: Path = Path(args.target)
    pdf: StructuredPDF = StructuredPDF()
    pdf.from_json(jsonpath)

    _print_content_of_pdf(pdf)


def retry_handler():
    pklpath: Path = Path(args.target)
    outdir: Path = Path(args.output)
    with open(pklpath, 'rb') as f:
        pdfextractor: PDFExtractor = pickle.load(f)

    idss: list[list[int]] = []
    while True:
        retry_ids = input('Which entry you want to retry? Input: \n')
        if retry_ids == 'ok': break
        idss.append([int(each) for each in retry_ids.split('.')])

    asyncio.run(pdfextractor._retry_entry(idss))

    with open(outdir / (pklpath.stem + '.json'), 'wt', encoding = 'utf-8') as f:
        f.write(json.dumps(pdfextractor.data.data, indent = 4, ensure_ascii = False))
    with open(outdir / (pklpath.stem + '.pkl'), 'wb') as f:
        pickle.dump(pdfextractor, f)
    simpidlog.info('Done.')

def export_handler() -> None:
    jsonpath: Path = Path(args.target)
    outdir: Path = Path(args.output)

    pdf: StructuredPDF = StructuredPDF()
    pdf.from_json(jsonpath = jsonpath)


    id_stack: list[int] = []
    def layer_iterator(id_stack: list[int], outdir: Path, pdf: list[dict], all_text: list[str]):
        for entry in pdf:
            id: int = entry['id']
            title: str = entry['title']
            text: str = entry['text']
            childs: list[dict] = entry['childs']

            id_stack.append(id)

            if len(text) > 1:
                with open(outdir / f'{'.'.join(list(map(str, id_stack)))}_{title}.md', 'wt', encoding = 'utf-8') as f:
                    f.write(text)
                all_text.append(text)
            if childs:
                layer_iterator(id_stack, outdir, childs, all_text)
            id_stack.pop(-1)

    all_text: list[str] = []
    layer_iterator([], outdir, pdf.data, all_text)

    with open(outdir / (str(Path(args.target).stem) + '.txt'), 'wt', encoding = 'utf-8') as f:
        f.write('\n'.join(all_text))



def main():
    cmd: str = args.cmd

    match cmd:
        case 'extract':
            extract_handler()
        case 'print':
            print_handler()
        case 'retry':
            retry_handler()
        case 'export':
            export_handler()


if __name__ == '__main__':
    simpidlog.set_basedir(args.logdir)
    simpidlog.set_whether_synchronous(True)
    simpidlog.start()

    try:
        main()
    finally:
        simpidlog.wait_for_log_io()
