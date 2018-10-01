import numpy as np
import os.path


def periodic_generator(L, p, batch_size, rn_level = 2):
    """
    Generate training data for toric code. (not very efficient)
    :param L: lattice size (code distance)
    :param p: bit-flip error rate
    :param batch_size: batch size
    :param rn_level: size of unit cell. Notably, if rn_level = L, the output of this generator contains
                     logical correction
    :return:
    """

    while True:
        syndrome = np.zeros((batch_size, L, L), dtype=np.bool)

        cg_err = np.zeros((batch_size, L // rn_level, L // rn_level, 2), dtype=np.bool)
        # x(e) for coarse-grained edge e, see Section II.B of the paper

        for y in range(L):
            for x in range(L):

                # first we randomly decide whether to place a bit flip error to the right
                error = np.random.choice(np.array([False, True]), size=batch_size, p=[1 - p, p])

                # if an error happen, (x,y) and (x,y+1) of syndrome is flipped
                syndrome[:, x, y] = np.logical_xor(error, syndrome[:, x, y])
                yp = (y + 1) % L  # periodic boundary condition
                syndrome[:, x, yp] = np.logical_xor(error, syndrome[:, x, yp])

                # update cg_err
                if (y + 1) % rn_level == 0:
                    cg_err[:, x // rn_level, yp // rn_level, 0] = \
                        np.logical_xor(error, cg_err[:, x // rn_level, yp // rn_level, 0])

                # now the same for a bit flip to the bottom
                error = np.random.choice(np.array([False, True]), size=batch_size, p=[1 - p, p])

                syndrome[:, x, y] = np.logical_xor(error, syndrome[:, x, y])

                xp = (x + 1) % L
                syndrome[:, xp, y] = np.logical_xor(error, syndrome[:, xp, y])

                if (x + 1) % rn_level == 0:
                    cg_err[:, xp // rn_level, y // rn_level, 1] = \
                        np.logical_xor(error, cg_err[:, xp // rn_level, y // rn_level, 1])

        yield (syndrome.reshape((batch_size, L, L, 1)).astype(np.int8), cg_err.astype(np.int8))


def pg_var_error_rate(L,  batch_size, rn_level = 2, min_p = 0.00, max_p = 0.16, np_file ='var_error_rate_000_016_L16.npy'):

    if os.path.isfile(np_file):
        p_error = np.load(np_file)
    else:
        p_error = np.ones((1, L, L, 2))*min_p + np.random.randint(2,size=(1, L, L, 2))*(max_p-min_p)
        np.save(np_file, p_error)

    p_error = np.broadcast_to(p_error,(batch_size,L,L,2))

    # From here it's very similar to periodic_generator
    while True:
        syndrome = np.zeros((batch_size, L, L), dtype=np.bool)

        cg_err = np.zeros((batch_size, L // rn_level, L // rn_level, 2), dtype=np.bool)

        for y in range(L):
            for x in range(L):

                yp = (y + 1) % L
                xp = (x + 1) % L

                temp = np.random.rand(batch_size)
                error = p_error[:,x,yp,0] > temp

                syndrome[:, x, y] = np.logical_xor(error, syndrome[:, x, y])
                syndrome[:, x, yp] = np.logical_xor(error, syndrome[:, x, yp])

                if (y + 1) % rn_level == 0:
                    cg_err[:, x // rn_level, yp // rn_level, 0] = \
                        np.logical_xor(error, cg_err[:, x // rn_level, yp // rn_level, 0])


                temp = np.random.rand(batch_size)
                error = p_error[:, xp, y, 1] > temp

                syndrome[:, x, y] = np.logical_xor(error, syndrome[:, x, y])
                syndrome[:, xp, y] = np.logical_xor(error, syndrome[:, xp, y])

                if (x + 1) % rn_level == 0:
                    cg_err[:, xp // rn_level, y // rn_level, 1] = \
                        np.logical_xor(error, cg_err[:, xp // rn_level, y // rn_level, 1])

        yield (syndrome.reshape((batch_size, L, L, 1)).astype(np.int8), cg_err.astype(np.int8), p_error)


def pg4bp(L,  batch_size, rn_level = 2, uniform_p = None):
    """
    Generator related to belief propagation data generation. Will add documentation later
    :param L:
    :param batch_size:
    :param rn_level:
    :param uniform_p:
    :return:
    """

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

