import sys, re
muts = {
 'addr4': ('	return fmt.Sprintf("(%s)(%s*4)", base, s)\n}', '	return fmt.Sprintf("(%s)(%s*1)", base, s3)\n}'),
 'noblend': ('				e.twStore(r, j, 12, 13, 14, 15, first)', '				e.twStore(r, j, 12, 13, 14, 15, false)'),
 'y1y4': ('	return [5]int{7, 1, 2, 3, 4}', '	return [5]int{7, 4, 2, 3, 1}'),
 'bfly3swap': ('			e.comp12Bfly3([3]int{0, 1, 2}, [3]int{3, 4, 5}, [3]int{6, 7, 8})\n			z = []int{3, 4, 5}', '			e.comp12Bfly3([3]int{0, 1, 2}, [3]int{3, 4, 5}, [3]int{6, 7, 8})\n			z = []int{3, 5, 4}'),
 'knext': ('		Raw("LEAQ (%s)(CX*4), AX", comp2InBase[q-1]).', '		Raw("LEAQ (%s)(CX*2), AX", comp2InBase[q-1]).'),
 'y0assoc': ('	e.add(7, 0, 3)\n	e.add(7, 7, 5) // y0', '	e.add(7, 3, 5)\n	e.add(7, 0, 7) // y0'),
 'q2sub': ('			e.add(2, 0, 1)\n			e.sub(3, 0, 1)\n			z = []int{2, 3}', '			e.add(2, 0, 1)\n			e.sub(3, 1, 0)\n			z = []int{2, 3}'),
 'lastadv': ('	b.Raw("ADDQ $%d, AX", 32*r)\n	for _, o := range comp2OutBase[:q] {', '	b.Raw("ADDQ $%d, AX", 16*r)\n	for _, o := range comp2OutBase[:q] {'),
}
name = sys.argv[1]; p = sys.argv[2]
s = open(p).read()
old, new = muts[name]
assert s.count(old) == 1, (name, s.count(old))
open(p, 'w').write(s.replace(old, new))
