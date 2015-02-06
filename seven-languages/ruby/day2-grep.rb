if ARGV.length < 2
   p "Usage day2-grep.rb FILENAME PATTERN"
end

pattern = ARGV[1]

infile = File.open(ARGV[0])
infile.each_line.with_index do |line, nr|
   if (/#{pattern}/ =~ line)
      puts "#{nr}: #{line}"
   end
end
infile.close()
