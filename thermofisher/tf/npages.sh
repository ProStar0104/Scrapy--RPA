tac tf.out | sed -n -r -e 's/.*INFO: Crawled ([[:digit:]]+) pages.*/\1/p' | head -1
