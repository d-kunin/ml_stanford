file = open('/Users/7times6/temp/lines.txt', 'r')
arg = ''
for line in file.readlines():
    arg += line.strip() + '\\n'
    
#print arg

to = open("args.txt", 'w')
to.write(arg)
to.close()