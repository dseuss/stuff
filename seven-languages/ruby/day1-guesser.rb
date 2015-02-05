def str_to_int_safe(i)
   return Integer(i)
rescue
   return nil
end


target = rand(10)

begin
   puts "Please enter a number from 0..10"
   guess = str_to_int_safe(gets)

   unless guess == nil
      puts "Lower" if guess > target
      puts "Higher" if guess < target
   else
      puts "That's not even a number, dummy"
   end
end while guess != target

puts "Good!"
