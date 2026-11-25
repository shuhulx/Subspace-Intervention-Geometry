# Third-party material

The experiment uses data and methods from [The Pain Axis](https://arxiv.org/abs/2609.16247v3), with the [original source](https://github.com/valen-research/Pain-axis/tree/213fb6c89c96004dbe93d1704c636ad8c9d801f3) and the joint projection correction from [PR #4](https://github.com/valen-research/Pain-axis/pull/4). Credit for the mathematical correction belongs to that upstream contribution.

The upstream repository declares the MIT license, reproduced below. Upstream-derived prompts and material retain their attribution. The full upstream scripts and model weights are not bundled; the local experiment loads selected functions from separately obtained, pinned upstream files.

Gemma 2 2B Instruct is provided under its [model terms](https://huggingface.co/google/gemma-2-2b-it). This repository includes generated continuations and the two pilot direction arrays recovered from an archived JSON record. It does not distribute the model weights or the original serialized direction tensor file. See [provenance.json](provenance.json) for the recovered record's source and content identity.

A project license has not yet been assigned to the original pilot code, documentation, annotations, or derived records. The upstream MIT notice covers upstream material; it should not be read as a blanket license for every file here. The Apache-2.0 and CC BY 4.0 licenses in ConceptorComplementGain do not automatically apply to this separate repository.

## Upstream MIT license

MIT License

Copyright (c) 2026 valen-research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
