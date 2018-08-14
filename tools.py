import numpy as np
import math, os.path

import pickle


class OnlineBatchGenerator:
    def __init__(self, L, p, batch_size):
        self.L = L
        self.p = p
        self.batch_size = batch_size

    def next_batch(self, rn_level=0):

        anyons = np.zeros((self.batch_size, self.L, self.L), dtype=np.bool)
        logicals = np.zeros((self.batch_size), dtype=np.bool)

        if rn_level > 0:
            rn_level = 2 ** rn_level
            rn_current = np.zeros((self.batch_size, self.L // rn_level, self.L // rn_level, 2), dtype=np.bool)

        for y in range(self.L):
            for x in range(self.L):

                # first we randomly decide whether to place a bit flip error to the right

                error = np.random.choice(np.array([False, True]), size=self.batch_size, p=[1 - self.p, self.p])

                #  syndrome is flipped at (x,y) and (x+1,y)
                # print(anyons[:,x,y].shape, error.shape)
                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])

                # if y+1 is out of range, the anyon goes off the right edge and is unrecorded
                if ((y + 1) < self.L):
                    anyons[:, x, y + 1] = np.logical_xor(error, anyons[:, x, y + 1])
                    if rn_level > 0:
                        if (y + 1) % rn_level == 0:
                            rn_current[:, x // rn_level, y // rn_level + 1, 0] = np.logical_xor(error, rn_current[:,
                                                                                                       x // rn_level,
                                                                                                       y // rn_level + 1,
                                                                                                       0])
                            # the third co-ordinate 0 corresponds to the left edge

            # no possibility to go off edge in this direction, so loop ensures x+1<L
            for x in range(self.L - 1):

                # now the same for a bit flip to the bottom
                error = np.random.choice(np.array([False, True]), size=self.batch_size, p=[1 - self.p, self.p])
                #  syndrome is flipped at (x,y) and (x,y+1)
                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])
                anyons[:, x + 1, y] = np.logical_xor(error, anyons[:, x + 1, y])

                if rn_level > 0:
                    if (x + 1) % rn_level == 0:
                        rn_current[:, x // rn_level, y // rn_level, 1] = np.logical_xor(error,
                                                                                        rn_current[:, x // rn_level,
                                                                                        y // rn_level, 1])
                        # the third co-ordinate 1 corresponds to the down edge

        # not all errors have yet been done. loop over the left edge
        for x in range(self.L):
            # add errors across edge in the same way as above
            error = np.random.choice(np.array([False, True]), size=self.batch_size, p=[1 - self.p, self.p])
            anyons[:, x, 0] = np.logical_xor(error, anyons[:, x, 0])
            # errors across edge in this case are logical errors, and are recorded as such

            if rn_level > 0:
                rn_current[:, x // rn_level, 0, 0] = np.logical_xor(error, rn_current[:, x // rn_level, 0, 0])
            logicals = np.logical_xor(logicals, error)

        if rn_level > 0:
            return anyons.reshape((self.batch_size, self.L, self.L, 1)).astype(np.float32), \
                   rn_current.astype(np.float32), logicals.reshape(self.batch_size, 1).astype(np.float32)
        else:
            return anyons.reshape((self.batch_size, self.L, self.L, 1)).astype(np.float32), \
                   logicals.reshape(self.batch_size, 1).astype(np.float32)

    def next_batch_al(self):

        tot_rn_lvl = int(math.log(self.L, 2))

        anyons = np.zeros((self.batch_size, self.L, self.L), dtype=np.bool)
        logicals = np.zeros((self.batch_size), dtype=np.bool)

        rn_current = {}

        for i in range(1, tot_rn_lvl):
            rn_level = 2 ** i
            rn_current[i - 1] = np.zeros((self.batch_size, self.L // rn_level, self.L // rn_level, 2), dtype=np.bool)

        for y in range(self.L):
            for x in range(self.L):

                # first we randomly decide whether to place a bit flip error to the right

                error = np.random.choice(np.array([False, True]), size=self.batch_size, p=[1 - self.p, self.p])

                #  syndrome is flipped at (x,y) and (x+1,y)
                # print(anyons[:,x,y].shape, error.shape)
                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])

                # if y+1 is out of range, the anyon goes off the right edge and is unrecorded
                if ((y + 1) < self.L):
                    anyons[:, x, y + 1] = np.logical_xor(error, anyons[:, x, y + 1])
                    for i in range(1, tot_rn_lvl):
                        rn_level = 2 ** i
                        if (y + 1) % rn_level == 0:
                            rn_current[i - 1][:, x // rn_level, y // rn_level + 1, 0] = np.logical_xor(error,
                                                                                                       rn_current[
                                                                                                           i - 1][:,
                                                                                                       x // rn_level,
                                                                                                       y // rn_level + 1,
                                                                                                       0])
                            # the third co-ordinate 0 corresponds to the left edge

            # no possibility to go off edge in this direction, so loop ensures x+1<L
            for x in range(self.L - 1):

                # now the same for a bit flip to the bottom
                error = np.random.choice(np.array([False, True]), size=self.batch_size, p=[1 - self.p, self.p])
                #  syndrome is flipped at (x,y) and (x,y+1)
                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])
                anyons[:, x + 1, y] = np.logical_xor(error, anyons[:, x + 1, y])

                for i in range(1, tot_rn_lvl):
                    rn_level = 2 ** i
                    if (x + 1) % rn_level == 0:
                        rn_current[i - 1][:, x // rn_level, y // rn_level, 1] = np.logical_xor(error,
                                                                                               rn_current[i - 1][:,
                                                                                               x // rn_level,
                                                                                               y // rn_level, 1])
                        # the third co-ordinate 1 corresponds to the down edge

        # not all errors have yet been done. loop over the left edge
        for x in range(self.L):
            # add errors across edge in the same way as above
            error = np.random.choice(np.array([False, True]), size=self.batch_size, p=[1 - self.p, self.p])
            anyons[:, x, 0] = np.logical_xor(error, anyons[:, x, 0])
            # errors across edge in this case are logical errors, and are recorded as such

            for i in range(1, tot_rn_lvl):
                rn_level = 2 ** i
                rn_current[i - 1][:, x // rn_level, 0, 0] = np.logical_xor(error,
                                                                           rn_current[i - 1][:, x // rn_level, 0, 0])
            logicals = np.logical_xor(logicals, error)

        return anyons.reshape((self.batch_size, self.L, self.L, 1)), rn_current, logicals.reshape(
            self.batch_size, 1)


def gen_dataset(size, p):
    batch_gen = OnlineBatchGenerator(32, p, size)
    anyons, rn_current, logicals = batch_gen.next_batch_al()

    with open('surf_training_32_' + str(p) + '_' + str(size), 'wb') as f:
        pickle.dump({'anyons': anyons, 'rn_current': rn_current, 'logicals': logicals}, f)


def periodic_generator(L, p, batch_size, rn_level = 2):

    while True:
        anyons = np.zeros((batch_size, L, L), dtype=np.bool)
        # logicals = np.zeros((batch_size), dtype=np.bool)


        rn_current = np.zeros((batch_size, L // rn_level, L // rn_level, 2), dtype=np.bool)

        for y in range(L):
            for x in range(L):

                # first we randomly decide whether to place a bit flip error to the right

                error = np.random.choice(np.array([False, True]), size=batch_size, p=[1 - p, p])

                #  syndrome is flipped at (x,y) and (x+1,y)
                # print(anyons[:,x,y].shape, error.shape)
                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])

                # if y+1 is out of range, the anyon goes off the right edge and is unrecorded

                yp = (y + 1) % L

                anyons[:, x, yp] = np.logical_xor(error, anyons[:, x, yp])

                if (y + 1) % rn_level == 0:
                    rn_current[:, x // rn_level, yp // rn_level, 0] = np.logical_xor(error, rn_current[:,
                                                                                            x // rn_level,
                                                                                            yp // rn_level,
                                                                                            0])

                # now the same for a bit flip to the bottom
                error = np.random.choice(np.array([False, True]), size=batch_size, p=[1 - p, p])

                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])

                xp = (x + 1) % L
                anyons[:, xp, y] = np.logical_xor(error, anyons[:, xp, y])

                if (x + 1) % rn_level == 0:
                    rn_current[:, xp // rn_level, y // rn_level, 1] = \
                        np.logical_xor(error,
                                       rn_current[:, xp // rn_level,
                                       y // rn_level, 1])

        yield (anyons.reshape((batch_size, L, L, 1)).astype(np.int8), rn_current.astype(np.int8))

    # if rn_level > 0:
    #     return anyons.reshape((batch_size, L, L, 1)).astype(np.float32), \
    #            rn_current.astype(np.float32), logicals.reshape(batch_size, 1).astype(np.float32)
    # else:
    #     return anyons.reshape((batch_size, L, L, 1)).astype(np.float32), \
    #            logicals.reshape(batch_size, 1).astype(np.float32)
    #


def pg4bp(L,  batch_size, rn_level = 2, uniform_p = None):

    while True:
        anyons = np.zeros((batch_size, L, L), dtype=np.bool)
        if uniform_p is not None:
            p_error = np.ones((batch_size, L, L,2)) * uniform_p
        else:
            p_error = np.exp(np.random.uniform(-7,-0.7, size=(batch_size, L, L,2)))
        # p_error = np.random.randint(2,size=(batch_size, L, L,2))

        rn_current = np.zeros((batch_size, L // rn_level, L // rn_level, 2), dtype=np.bool)

        for y in range(L):
            for x in range(L):

                # first we randomly decide whether to place a bit flip error to the right
                yp = (y + 1) % L
                xp = (x + 1) % L

                temp = np.random.rand(batch_size)
                error = p_error[:,x,yp,0] > temp

                #  syndrome is flipped at (x,y) and (x+1,y)
                # print(anyons[:,x,y].shape, error.shape)
                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])

                # if y+1 is out of range, the anyon goes off the right edge and is unrecorded

                anyons[:, x, yp] = np.logical_xor(error, anyons[:, x, yp])

                if (y + 1) % rn_level == 0:
                    rn_current[:, x // rn_level, yp // rn_level, 0] = np.logical_xor(error, rn_current[:,
                                                                                            x // rn_level,
                                                                                            yp // rn_level,
                                                                                            0])

                # now the same for a bit flip to the bottom
                temp = np.random.rand(batch_size)
                error = p_error[:, xp, y, 1] > temp

                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])
                anyons[:, xp, y] = np.logical_xor(error, anyons[:, xp, y])

                if (x + 1) % rn_level == 0:
                    rn_current[:, xp // rn_level, y // rn_level, 1] = \
                        np.logical_xor(error,
                                       rn_current[:, xp // rn_level,
                                       y // rn_level, 1])

        yield (anyons.reshape((batch_size, L, L, 1)).astype(np.int8), rn_current.astype(np.int8), p_error)


def pg_var_error_rate(L,  batch_size, rn_level = 2, min_p = 0.06, max_p = 0.1, np_file ='var_error_rate_L16.npy'):

    if os.path.isfile(np_file):
        p_error = np.load(np_file)
    else:
        p_error = np.random.uniform(min_p, max_p, size=(1, L, L, 2))
        np.save(np_file, p_error)

    p_error = np.broadcast_to(p_error,(batch_size,L,L,2))


    while True:
        anyons = np.zeros((batch_size, L, L), dtype=np.bool)

        rn_current = np.zeros((batch_size, L // rn_level, L // rn_level, 2), dtype=np.bool)

        for y in range(L):
            for x in range(L):

                # first we randomly decide whether to place a bit flip error to the right
                yp = (y + 1) % L
                xp = (x + 1) % L

                temp = np.random.rand(batch_size)
                error = p_error[:,x,yp,0] > temp

                #  syndrome is flipped at (x,y) and (x+1,y)
                # print(anyons[:,x,y].shape, error.shape)
                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])

                # if y+1 is out of range, the anyon goes off the right edge and is unrecorded

                anyons[:, x, yp] = np.logical_xor(error, anyons[:, x, yp])

                if (y + 1) % rn_level == 0:
                    rn_current[:, x // rn_level, yp // rn_level, 0] = np.logical_xor(error, rn_current[:,
                                                                                            x // rn_level,
                                                                                            yp // rn_level,
                                                                                            0])

                # now the same for a bit flip to the bottom
                temp = np.random.rand(batch_size)
                error = p_error[:, xp, y, 1] > temp

                anyons[:, x, y] = np.logical_xor(error, anyons[:, x, y])
                anyons[:, xp, y] = np.logical_xor(error, anyons[:, xp, y])

                if (x + 1) % rn_level == 0:
                    rn_current[:, xp // rn_level, y // rn_level, 1] = \
                        np.logical_xor(error,
                                       rn_current[:, xp // rn_level,
                                       y // rn_level, 1])

        yield (anyons.reshape((batch_size, L, L, 1)).astype(np.int8), rn_current.astype(np.int8), p_error)