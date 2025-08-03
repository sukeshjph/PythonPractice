while True:
    try:
        int('jjj')
        print('Try')

    except Exception as e:
        print(f"The error is {e}")
        break


try:
  b = 6
  if b > 4:
    raise ValueError('No out of range')

except Exception as e:
    print(e)




