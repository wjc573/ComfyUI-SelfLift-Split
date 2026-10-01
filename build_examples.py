"""Create two non-destructive editor-format H3 SelfLift Split examples."""
import copy
import json
from pathlib import Path

SOURCE = Path('E:/AI_painting/workflows/【视频工作流】')
TARGET = Path(__file__).resolve().parent / 'workflows'
NAMES = (
    'MiniMax H3 latent放大双采+8步lora 顶级质量和速度 全能参考工作流.json',
    'MiniMax H3 双采+8步lora 顶级质量和速度 全能参考工作流1.json',
)


def generate(source, learned):
    graph = json.loads(source.read_text(encoding='utf-8'))
    nodes = {node['id']: node for node in graph['nodes']}
    drop = {125, 156, 173, 188, 189, 190, 191, 192, 218, 219}
    graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in drop]
    graph['links'] = [link for link in graph['links'] if link[1] not in drop and link[3] not in drop]
    graph['groups'] = [group for group in graph.get('groups', []) if group['id'] == 9 or group['id'] not in (7, 9)]
    for group in graph['groups']:
        if group['id'] == 9:
            group['title'] = 'SelfLift Split / external latent lift / prepared AV resume'
            group['bounding'] = [440, 4810, 1050, 720]
    nodes[123]['widgets_values'][0] = 'euler'
    nodes[115]['widgets_values'] = ['9:16 (Portrait Widescreen)', 0.8, 32]  # final canvas ~0.8 MP (~1024x1792px at 9:16)
    nodes[136]['widgets_values'][1:3] = [1024, 1792]  # unconnected widget fallback for conditioning
    nodes[175]['pos'] = [1630, 5000]
    nodes[126]['pos'] = [1400, 5560]
    graph['nodes'].append(nodes[126]) if nodes[126] not in graph['nodes'] else None
    nodes[126]['outputs'][0]['links'] = [l[0] for l in graph['links'] if l[1] == 126]
    max_id = max(n['id'] for n in graph['nodes'])
    max_link = max((l[0] for l in graph['links']), default=0)

    def add_node(node_type, inputs, outputs, pos, widgets=None, size=None):
        nonlocal max_id
        max_id += 1
        n = {'id': max_id, 'type': node_type, 'pos': pos, 'size': size or [290, 160],
             'flags': {}, 'order': 100 + max_id, 'mode': 0, 'inputs':
             [{'name': name, 'type': typ, 'link': None} for name, typ in inputs],
             'outputs': [{'name': name, 'type': typ, 'links': []} for name, typ in outputs],
             'properties': {'Node name for S&R': node_type}, 'widgets_values': widgets or []}
        graph['nodes'].append(n)
        nodes[n['id']] = n
        return n

    def connect(source_id, source_slot, target_id, target_slot):
        nonlocal max_link
        max_link += 1
        src, dst = nodes[source_id], nodes[target_id]
        old = dst['inputs'][target_slot].get('link')
        if old:
            disconnect(old)
        typ = dst['inputs'][target_slot]['type']
        graph['links'].append([max_link, source_id, source_slot, target_id, target_slot, typ])
        src['outputs'][source_slot]['links'] = (src['outputs'][source_slot].get('links') or []) + [max_link]
        dst['inputs'][target_slot]['link'] = max_link

    def disconnect(link_id):
        old = next(l for l in graph['links'] if l[0] == link_id)
        src = nodes[old[1]]
        src['outputs'][old[2]]['links'].remove(link_id)
        graph['links'].remove(old)

    # Remove orphaned old sampling connections and preserve all peripheral links.
    for node in graph['nodes']:
        for inp in node.get('inputs', []):
            if inp.get('link') not in {l[0] for l in graph['links']}:
                inp['link'] = None
        for out in node.get('outputs', []):
            out['links'] = [i for i in (out.get('links') or []) if i in {l[0] for l in graph['links']}]

    low = add_node('SelfLiftSplitLowH3',
                   [('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT'),('sampler','SAMPLER'),('sigmas','SIGMAS'),('seed','INT'),('cfg','FLOAT'),('transition_step','INT'),('lowres_scale','FLOAT')],
                   [('state','SELF_LIFT_SPLIT_STATE'),('clean_video','LATENT'),('positive_low','CONDITIONING'),('negative_low','CONDITIONING'),('target_width_pixels','INT'),('target_height_pixels','INT')],
                   [170, 4890], [1., .5], [330, 340])
    transition = add_node('SelfLiftSplitTransitionH3',
                          [('state','SELF_LIFT_SPLIT_STATE'),('upscaled_video','LATENT'),('model_high','MODEL'),('mode','COMBO'),('rho','FLOAT'),('w_min','FLOAT'),('w_max','FLOAT'),('vae','VAE')],
                          [('resume_latent','LATENT'),('tail_sigmas','SIGMAS'),('model_high','MODEL')],
                          [1070, 4900], ['legacy_seed' if learned else 'simple_interpolation', 0., .5, 1.], [305, 280])
    seed_node = add_node('PrimitiveInt', [('value','INT')], [('INT','INT')], [-260, 4860], [nodes[129]['widgets_values'][0], 'fixed'], [160, 52])
    for source_id, slot, target_slot in [(146,0,0),(136,0,1),(136,0,2),(136,1,3),(123,0,4),(124,0,5),
                                         (seed_node['id'],0,6),(185,0,8)]:
        connect(source_id, slot, low['id'], target_slot)
    connect(low['id'], 0, transition['id'], 0)
    connect(146, 0, transition['id'], 2)
    if learned:
        up = add_node('MinimaxH3LatentUpscaler3D',
                      [('latent','*'),('mode.width','INT'),('mode.height','INT')],
                      [('latent','*')], [620, 4880],
                      ['minimax_h3_latent_upscaler_3d_bf16.safetensors', 'target dimensions', 1024, 1792, 32, True, True, 'rocm', 'bf16'], [405, 240])
        connect(low['id'], 1, up['id'], 0)
        connect(low['id'], 4, up['id'], 1)
        connect(low['id'], 5, up['id'], 2)
    else:
        up = add_node('SelfLiftSplitResizeH3',
                      [('state','SELF_LIFT_SPLIT_STATE'),('video','LATENT'),('method','COMBO')],
                      [('LATENT','LATENT')], [640, 4900], ['nearest'], [320, 120])
        connect(low['id'], 0, up['id'], 0)
        connect(low['id'], 1, up['id'], 1)
    connect(up['id'], 0, transition['id'], 1)
    connect(119, 0, transition['id'], 7)
    connect(transition['id'], 0, 175, 4)
    connect(transition['id'], 1, 175, 3)
    connect(transition['id'], 2, 126, 0)
    graph['last_node_id'] = max_id
    graph['last_link_id'] = max_link
    return graph


def validate(g):
    nodes = {node['id']: node for node in g['nodes']}
    links = {l[0]: l for l in g['links']}
    assert len(nodes) == len(g['nodes']) and len(links) == len(g['links'])
    for link in links.values():
        _, start, slot, end, inp, typ = link
        assert start in nodes and end in nodes, link
        assert nodes[start]['outputs'][slot]['links'].count(link[0]) == 1, link
        assert nodes[end]['inputs'][inp]['link'] == link[0], link
        assert typ in nodes[end]['inputs'][inp]['type'].split(','), link
    for node in nodes.values():
        for inp in node.get('inputs', []):
            assert inp.get('link') is None or inp['link'] in links, (node['id'], inp)
        for out in node.get('outputs', []):
            assert all(i in links for i in (out.get('links') or [])), (node['id'], out)
    assert nodes[175]['inputs'][0]['link'] in links and nodes[175]['inputs'][4]['link'] in links
    return len(nodes), len(links)


def main():
    TARGET.mkdir(exist_ok=True)
    for name, learned in zip(NAMES, (True, False)):
        graph = generate(SOURCE / name, learned)
        count = validate(graph)
        path = TARGET / ('H3-SelfLift-Split-' + ('learned-beta' if learned else 'simple-interpolation') + '.json')
        path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding='utf-8')
        print(path, count)


if __name__ == '__main__':
    main()
